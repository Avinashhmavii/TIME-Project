import os
import json
import time
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from openai import OpenAI
import pdfplumber
import docx2txt
from dotenv import load_dotenv
from collections import defaultdict
import logging
import re
import threading
import cv2
import numpy as np
from datetime import datetime
import base64
import platform

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = os.urandom(24)
os.makedirs('uploads', exist_ok=True)
os.makedirs('uploads/snapshots', exist_ok=True)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not found in .env. OpenAI dependent features will not work.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    client = None

qna_evaluations = []
current_use_voice_mode = False
listening_active = False
interview_context = {}
visual_analysis_thread = None
visual_analyses = []

interview_context_template = {
    'questions_list': [], 'current_q_idx': 0, 'previous_answers_list': [], 'scores_list': [],
    'question_depth_counter': 0, 'max_followup_depth': 2, 'current_interview_track': None,
    'current_sub_track': None, 'questions_already_asked': set(), 'current_job_description': None,
    'use_camera_feature': False,
    'generated_resume_questions_cache': [],
    'icebreaker_was_prepended': False,
    'prepended_icebreaker_text': None
}

structure = {
    'mba': {'resume_flow': [], 'school_based': defaultdict(list), 'interest_areas': defaultdict(list)},
    'bank': {'resume_flow': [], 'bank_type': defaultdict(list), 'technical_analytical': defaultdict(list)}
}
mba_pdf_path = "MBA_Question.pdf"
bank_pdf_path = "Bank_Question.pdf"

def normalize_text(text_input):
    if not text_input: return ""
    return " ".join(str(text_input).strip().split()).lower()

def strip_numbering(text_input):
    if not text_input: return ""
    return re.sub(r'^\d+\.\s*', '', str(text_input)).strip()

def load_questions_into_memory(pdf_path, section_type):
    if not os.path.exists(pdf_path):
        logging.error(f"PDF question file '{pdf_path}' not found.")
        return False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ''.join(page.extract_text() or '' for page in pdf.pages if page.extract_text())
        lines = full_text.split('\n')
        current_section = None
        current_subsection = None
        for line in lines:
            line = line.strip()
            if not line: continue
            if section_type == 'mba':
                if "1. Resume Flow" in line: current_section, current_subsection = 'resume_flow', None; continue
                elif "2. Pre-Defined Question Selection" in line: current_section, current_subsection = 'school_based', None; continue
                elif "3. Interface to Select Question Areas" in line: current_section, current_subsection = 'interest_areas', None; continue
                if current_section == 'school_based':
                    if "For IIMs" in line: current_subsection = 'IIM'; continue
                    elif "For ISB" in line: current_subsection = 'ISB'; continue
                    elif "For Other B-Schools" in line: current_subsection = 'Other'; continue
                if current_section == 'interest_areas':
                    if "General Business & Leadership" in line: current_subsection = 'General Business'; continue
                    elif "Finance & Economics" in line: current_subsection = 'Finance'; continue
                    elif "Marketing & Strategy" in line: current_subsection = 'Marketing'; continue
                    elif "Operations & Supply Chain" in line: current_subsection = 'Operations'; continue
            elif section_type == 'bank':
                if "Resume-Based Questions" in line: current_section, current_subsection = 'resume_flow', None; continue
                elif "Bank-Type Specific Questions" in line: current_section, current_subsection = 'bank_type', None; continue
                elif "Technical & Analytical Questions" in line: current_section, current_subsection = 'technical_analytical', None; continue
                elif "Current Affairs" in line: current_section, current_subsection = 'technical_analytical', 'Current Affairs'; continue
                if current_section == 'bank_type':
                    if "Public Sector Banks" in line: current_subsection = 'Public Sector Banks'; continue
                    elif "Private Banks" in line: current_subsection = 'Private Banks'; continue
                    elif "Regulatory Roles" in line: current_subsection = 'Regulatory Roles'; continue
                if current_section == 'technical_analytical' and current_subsection != 'Current Affairs':
                    if "Banking Knowledge" in line: current_subsection = 'Banking Knowledge'; continue
                    elif "Logical Reasoning" in line: current_subsection = 'Logical Reasoning'; continue
                    elif "Situational Judgement" in line: current_subsection = 'Situational Judgement'; continue
            if line and line[0].isdigit() and '.' in line.split()[0]:
                question_text = strip_numbering(line)
                if not question_text: continue
                is_sequence = bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question_text))
                question_data = {'text': question_text, 'type': 'sequence' if is_sequence else 'standard'}
                if not question_data['text'].endswith('?'): question_data['text'] += '?'
                if current_section == 'resume_flow': structure[section_type]['resume_flow'].append(question_data)
                elif current_section and current_subsection: structure[section_type][current_section][current_subsection].append(question_data)
        logging.info(f"Successfully loaded questions for {section_type} from {pdf_path}.")
        return True
    except Exception as e_load:
        logging.error(f"Error loading questions from {pdf_path} for {section_type}: {e_load}", exc_info=True)
        return False

if not load_questions_into_memory(mba_pdf_path, 'mba'):
    logging.warning(f"Could not load MBA questions from '{mba_pdf_path}'. Using minimal fallback.")
    structure['mba']['resume_flow'] = [{'text': "Tell me about your background and why you are pursuing an MBA?", 'type': 'standard'}]
if not load_questions_into_memory(bank_pdf_path, 'bank'):
    logging.warning(f"Could not load Bank questions from '{bank_pdf_path}'. Using minimal fallback.")
    structure['bank']['resume_flow'] = [{'text': "Why are you interested in a career in the banking sector?", 'type': 'standard'}]

def get_openai_response_generic(prompt_messages, temperature=0.7, max_tokens=500, model_override=None):
    if not client:
        logging.error("OpenAI client not available for API call.")
        return "OpenAI client not available."
    try:
        chosen_model = model_override if model_override else "gpt-4o-mini"
        response = client.chat.completions.create(
            model=chosen_model, messages=prompt_messages, temperature=temperature, max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e_openai:
        logging.error(f"OpenAI API call error with model {chosen_model}: {e_openai}", exc_info=True)
        return f"Error: OpenAI API Call Failed - {e_openai}"

def capture_initial_frame_data_for_question():
    cap = None
    try:
        # Try multiple camera indices (0, 1, 2) to find a working device
        for index in range(3):
            logging.info(f"Attempting to open camera at index {index}")
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                logging.info(f"Camera opened successfully at index {index}")
                break
        else:
            logging.error("Icebreaker: Failed to open any webcam device after trying indices 0-2.")
            return None

        # Ensure camera is ready
        time.sleep(1.0)  # Increased delay for Linux compatibility
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning("Icebreaker: Could not capture initial frame.")
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        image_data_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_data_base64}"
    except Exception as e_capture:
        logging.error(f"Icebreaker: Exception during initial frame capture: {e_capture}", exc_info=True)
        if platform.system() == "Linux":
            logging.info("Running on Linux. Check camera permissions (e.g., /dev/video0) and ensure v4l2loopback or similar is configured.")
        return None
    finally:
        if cap:
            cap.release()
            logging.info("Camera released after initial frame capture.")

def generate_environment_icebreaker_question(image_data_url):
    if not client:
        logging.warning("Icebreaker: OpenAI client not available.")
        return "Your setup looks well-prepared. Are you ready to begin the interview?"
    if not image_data_url:
        logging.warning("Icebreaker: No image data provided, using fallback question.")
        return "Your setup looks well-prepared. Are you ready to begin the interview?"
    try:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": (
                "You are an interviewer conducting a formal interview. The candidate has enabled their camera. "
                "To begin the interaction smoothly, observe their professional presentation or a general, neutral aspect of their visible environment from this image. "
                "You can ask about something in their background (like a bookshelf, a plant, or artwork if it looks professional) or make a general positive observation about their setup or readiness. "
                "If you observe something distinctly professional about their attire (e.g., a formal jacket, a tie) or a neat hairstyle that contributes to a professional image, you can make a brief, positive, and non-detailed comment that leads to a simple icebreaker question. Frame it carefully to be about their preparedness or professional setting. "
                "If visible, you may make a brief, general comment on something professional in their attire (e.g., a blazer, shirt, or tie) or grooming that suggests preparedness."
                "Ask a single, brief, and formal icebreaker question. The question must be polite, non-intrusive, and strictly professional. "
                "Avoid any overly personal remarks or overly casual phrasing. Ensure the question is a complete sentence ending with a question mark. "
                "Examples: 'I notice an interesting bookshelf behind you; do you have a favorite professional read?', 'Your setup looks very professional. Are you comfortable and ready to start?', 'That's a smart background choice. Does it help you focus for calls like these?'. "
                "If commenting on attire/appearance, it should be very general and focused on professionalism, for example: 'You present very professionally. I hope you're feeling well-prepared for our discussion today?' "
                "Do not thank the candidate for attending"
                "The goal is a polite, initial engagement. If unsure, stick to the environment or general readiness."
            )},
            {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}}
        ]}]
        response_text = get_openai_response_generic(messages, temperature=0.6, max_tokens=75, model_override="gpt-4o-mini")
        if "Error" in response_text or "OpenAI client not available" in response_text:
            logging.error(f"Icebreaker: API/client error: {response_text}")
            return "Your setup looks well-prepared. Are you ready to begin the interview?"
        question = response_text.strip()
        if question and question.endswith('?') and 3 <= len(question.split()) <= 35:
            logging.info(f"Icebreaker: Generated formal question: {question}")
            return question
        else:
            logging.warning(f"Icebreaker: Generated question unsuitable: '{question}' (Words: {len(question.split()) if question else 0})")
            return "Your setup looks well-prepared. Are you ready to begin the interview?"
    except Exception as e_ice:
        logging.error(f"Icebreaker: Exception in generation: {e_ice}", exc_info=True)
        return "Your setup looks well-prepared. Are you ready to begin the interview?"

def generate_resume_questions(resume_text, job_type, asked_qs_set_normalized_global):
    if not resume_text or resume_text == "Resume content appears to be empty or could not be extracted.":
        return ["Could you start by telling me a bit about your background and what led you to apply?"]
    prompt_context = "an MBA program interview" if job_type == 'mba' else "a banking role interview"
    prompt = (
        f"You are an expert interviewer preparing for {prompt_context}. "
        f"Based only on the candidate's resume provided below, generate 10-12 unique, insightful questions. "
        f"Focus on their experiences, skills, achievements, and career progression as detailed in the resume. "
        f"Each question must be a complete sentence, concise, and end with a question mark. Avoid truncating questions mid-sentence."
        f"Do not ask generic questions not directly tied to the resume content. "
        f"interview questions tailored to the candidate's experience and background." 
        f"Avoid questions similar to these already considered (normalized sample): {list(asked_qs_set_normalized_global)[:3]}. "
        f"Resume Text: ```{resume_text[:2500]}```"
    )
    response_text = get_openai_response_generic([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.55)
    fallback_qs_list = [
        "Walk me through the key highlights of your resume.",
        "Which accomplishment on your resume are you most proud of, and what was your specific role?",
        "Can you elaborate on your responsibilities and achievements in your role at [mention a company/role from resume if discernible, else 'your most recent position']?"
    ]
    if "Error" in response_text or "OpenAI client not available" in response_text:
        logging.warning(f"Resume Q Gen: API/Client error: {response_text}"); return fallback_qs_list
    generated_qs_raw_list = [strip_numbering(q.strip()) for q in response_text.split('\n') if q.strip()]
    final_resume_qs = []
    for q_text_candidate in generated_qs_raw_list:
        if not q_text_candidate.endswith('?'): q_text_candidate += '?'
        if 3 <= len(q_text_candidate.split()) <= 30:
            if normalize_text(q_text_candidate) not in asked_qs_set_normalized_global:
                final_resume_qs.append(q_text_candidate)
    if len(final_resume_qs) >= 5: return final_resume_qs[:10]
    else:
        logging.info(f"Resume Q Gen: Only {len(final_resume_qs)} unique questions. Supplementing.")
        for f_q_text in fallback_qs_list:
            if normalize_text(f_q_text) not in {normalize_text(q) for q in final_resume_qs}: final_resume_qs.append(f_q_text)
        return final_resume_qs[:10]

def generate_answer_feedback(question, answer, job_description):
    prompt = f"""
You are an expert interviewer for a role related to: {job_description}.
The candidate was asked: "{question}"
Candidate's Answer: "{answer}"
provide concise, constructive feedback to help the candidate improve their interview performance. Focus on clarity, detail, relevance to the question, and communication skills. Provide 2-3 sentences of specific, actionable advice tailored to the answer's content and weaknesses. Avoid repeating the question or answer verbatim, and do not include scores or numerical ratings. Ensure the feedback is encouraging, professional, and unique for each response.
Feedback:"""
    feedback = get_openai_response_generic([{"role": "user", "content": prompt}], temperature=0.65, max_tokens=160)
    if "Error" in feedback or "OpenAI client not available" in feedback:
        return "Feedback could not be generated. Focus on clear, structured answers with specific examples."
    return feedback.strip()

CATEGORY_ALIASES_EVAL = {
    "ideas": "Ideas",
    "organization": "Organization",
    "accuracy": "Accuracy",
    "voice": "Voice",
    "grammar usage and sentence fluency": "Grammar Usage and Sentence Fluency",
    "stop words": "Stop words"
}
WEIGHTS_EVAL = {
    "Ideas": 0.2,
    "Organization": 0.25,
    "Accuracy": 0.2,
    "Voice": 0.2,
    "Grammar Usage and Sentence Fluency": 0.05,
    "Stop words": 0.1
}
def parse_evaluation_response(raw_response_text):
    parsed_eval = {}
    lines = [line.strip() for line in raw_response_text.split('\n') if line.strip()]
    current_category_key_eval = None
    for line in lines:
        match_eval = re.match(r'^Category:\s*(.+?)\s*\((\d{1,2})(?:/10)?\)$', line, re.IGNORECASE)
        if match_eval:
            category_name_raw_eval = match_eval.group(1).strip()
            canonical_name_eval = None
            for alias_key, canonical_val in CATEGORY_ALIASES_EVAL.items():
                if category_name_raw_eval.lower() == alias_key.lower() or category_name_raw_eval.lower() == canonical_val.lower():
                    canonical_name_eval = canonical_val; break
            if not canonical_name_eval: canonical_name_eval = category_name_raw_eval
            score_val = int(match_eval.group(2).strip())
            parsed_eval[canonical_name_eval] = {"score": score_val}
            current_category_key_eval = canonical_name_eval
            continue
        if current_category_key_eval and line.lower().startswith("justification:"):
            justification_text = line.split(":", 1)[1].strip()
            if current_category_key_eval in parsed_eval:
                 parsed_eval[current_category_key_eval]["justification"] = justification_text
            current_category_key_eval = None
    return parsed_eval

def calculate_weighted_evaluation_score(scores_dict_eval):
    total_weighted_score_val = 0.0; total_weight_applied_val = 0.0
    for category_name_eval, eval_values in scores_dict_eval.items():
        score_num = eval_values.get("score", 0)
        weight_val = WEIGHTS_EVAL.get(category_name_eval, 0)
        if weight_val > 0: total_weighted_score_val += score_num * weight_val; total_weight_applied_val += weight_val
    if total_weight_applied_val == 0: return 0.0
    return round(total_weighted_score_val, 2)

def evaluate_sequence_response(question_text, answer_text):
    if "2, 5, 10, 17, 26" in question_text.replace(" ", ""):
        correct_ans_seq = "37"
        user_ans_digits = re.findall(r'\d+', str(answer_text))
        if user_ans_digits and user_ans_digits[0] == correct_ans_seq: return "[Correct sequence. Well done!] Score: 10/10", 10
        else: return f"[Incorrect sequence. Expected {correct_ans_seq}. Your answer: {answer_text}] Score: 0/10", 0
    return "[Sequence pattern not specifically programmed. Assessed qualitatively.] Score: 5/10", 5

def fallback_ai_evaluation(question_text, answer_text):
    answer_norm = normalize_text(answer_text)
    if not answer_norm or len(answer_norm) < 10: return "[Answer too brief/empty. Unable to evaluate using fallback.] Score: 0/10", 0
    score_fb = min(10, max(2, len(answer_text.split()) // 4))
    feedback_fb = ("[Fallback Eval: Answer relevant. Consider more detail/structure.]" if score_fb < 7 else "[Fallback Eval: Answer relevant and reasonably detailed.]")
    return f"{feedback_fb} Score: {score_fb}/10", score_fb

def evaluate_response_with_ai_scoring(question_text, answer_text, job_description_context):
    if not answer_text or answer_text.strip() == "" or answer_text.lower() == "no answer provided by candidate.":
        return "[No effective answer provided for AI scoring.] Score: 0/10", 0
    if bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question_text)): return evaluate_sequence_response(question_text, answer_text)
    prompt_eval = f"""
You are an AI Interview Performance Analyzer for a {job_description_context} role.
Evaluate the candidate's answer to the question below based on these exact six categories.
1. Ideas:
The answer should focus on one clear idea, maintained throughout without tangents.

2. Organization:
Ideas should flow logically and cohesively.

3. Accuracy:
The answer should fully address all parts of the question.

4. Voice:
The answer should be unique and not generic.

5. Grammar Usage and Sentence Fluency:
The answer should use correct grammar and sentence structure.

6. Stop words:
Minimize filler words (e.g., uhh, ahh, ummm).

Provide a score (1-10, 1 lowest, 10 highest) for each category with a one-line justification.

Format the response exactly as:
Category: <category> (<score>/10)
Justification: <explanation>

List all six categories.
Question: {question_text}
Candidate's Answer: {answer_text}
"""
    try:
        ai_eval_text = get_openai_response_generic([{"role": "user", "content": prompt_eval}], temperature=0.35, max_tokens=500)
        if "Error:" in ai_eval_text or "OpenAI client not available" in ai_eval_text:
            logging.warning(f"AI Scoring: API/Client error. Using fallback. Error: {ai_eval_text}"); return fallback_ai_evaluation(question_text, answer_text)
        parsed_scores_from_ai = parse_evaluation_response(ai_eval_text)
        if not parsed_scores_from_ai or len(parsed_scores_from_ai) < 6:
            logging.warning(f"AI Scoring: Failed to parse categories. Response: '{ai_eval_text}'. Parsed: {parsed_scores_from_ai}. Using fallback."); return fallback_ai_evaluation(question_text, answer_text)
        final_weighted_score = calculate_weighted_evaluation_score(parsed_scores_from_ai)
        eval_details_for_record = ["[AI Detailed Scoring Complete]"]
        for cat_name_record, data_record in parsed_scores_from_ai.items():
            eval_details_for_record.append(f"{cat_name_record}: {data_record.get('score', 'N/A')}/10 ({data_record.get('justification', 'N/J')})")
        full_eval_details_str = " | ".join(eval_details_for_record) + f" | Final Weighted Score: {final_weighted_score}/10"
        return full_eval_details_str, final_weighted_score
    except Exception as e_ai_score:
        logging.error(f"AI Scoring: Exception: {e_ai_score}", exc_info=True); return fallback_ai_evaluation(question_text, answer_text)

def generate_next_question(prev_q_text, prev_ans_text, prev_score, interview_track_context, job_type_context, asked_qs_normalized_set_global, attempt_num=1):
    if attempt_num > 2: logging.info("Follow-up Gen: Max attempts reached."); return None
    focus_map = {
        'resume': 'candidate specific experiences, skills, or career goals mentioned in their resume or previous answer',
        'school_based': 'their academic motivations, reasons for choosing a particular school, or how their studies relate to career goals',
        'interest_areas': 'their passion for the chosen interest area, depth of knowledge, or practical application of their interests',
        'bank_type': 'their understanding of the specific bank type, customer service approaches, or relevant operational aspects',
        'technical_analytical': 'their technical banking knowledge, problem-solving abilities, or logical reasoning based on the previous answer'
    }
    focus_guidance = focus_map.get(interview_track_context, 'general relevance, impact, or lessons learned from their previous answer')
    prompt_fu = (
        f"You are an interviewer for a {job_type_context} candidate. They just answered a question. "
        f"Previous Question: \"{prev_q_text}\"\nCandidate's Answer: \"{prev_ans_text}\"\nThis answer was scored {prev_score}/10.\n"
        f"Based on this, generate ONE insightful follow-up question that delves deeper into their response, focusing on {focus_guidance}. "
        f"The follow-up should be natural, concise, a complete sentence, and end with a question mark. "
        f"Do NOT repeat the previous question or ask something generic if a specific follow-up is possible. "
        f"Avoid questions similar to these already considered (normalized sample): {list(asked_qs_normalized_set_global)[:3]}. Follow-up Question:"
    )
    fu_resp_text = get_openai_response_generic([{"role": "user", "content": prompt_fu}], max_tokens=110, temperature=0.6)
    default_fu_q = f"Can you give me a specific example related to your last point?"
    if "Error" in fu_resp_text or "OpenAI client not available" in fu_resp_text:
        logging.warning(f"Follow-up Gen: API/Client Error: {fu_resp_text}"); return default_fu_q if attempt_num == 1 else None
    fu_q_candidate = strip_numbering(fu_resp_text.strip())
    if not fu_q_candidate.endswith('?'): fu_q_candidate += '?'
    if not (3 <= len(fu_q_candidate.split()) <= 30):
        logging.info(f"Follow-up Gen: Candidate '{fu_q_candidate}' failed word count (Attempt {attempt_num})."); return default_fu_q if attempt_num == 1 else None
    norm_fu_q_candidate = normalize_text(fu_q_candidate)
    if norm_fu_q_candidate in asked_qs_normalized_set_global:
        logging.info(f"Follow-up Gen: Candidate '{fu_q_candidate}' already asked (Attempt {attempt_num}).")
        if attempt_num == 1: return generate_next_question(prev_q_text, prev_ans_text, prev_score, interview_track_context, job_type_context, asked_qs_normalized_set_global, attempt_num + 1)
        return None
    logging.info(f"Follow-up Gen: Generated question (Attempt {attempt_num}): {fu_q_candidate}"); return fu_q_candidate

def generate_conversational_reply(answer_text, job_type_context):
    sys_prompt_ack = (f"You are an engaging and human-like {'HR' if job_type_context == 'mba' else 'banking HR'} interviewer. "
                      f"The candidate has just finished their answer. Generate a short, complete sentence as a reply. "
                      f"Your reply should be engaging and human-like, providing feedback or encouragement without asking for further information. "
                      f"Ensure it's a full thought. The reply MUST be a statement (ending with a period or exclamation mark) and MUST NOT contain any questions (do not end with a question mark). "
                      f"Examples: 'That's a very insightful way to put it.', 'I appreciate you sharing that experience with such clarity!', 'Excellent point, that really highlights your skills.', 'Thanks for that detailed explanation.'")
    ans_summary_for_prompt = answer_text[:100] + ("..." if len(answer_text) > 100 else "")
    ack_resp_text = get_openai_response_generic(
        [{"role": "system", "content": sys_prompt_ack}, {"role": "user", "content": f"Candidate's answer (summary): {ans_summary_for_prompt}"}],
        temperature=0.75, max_tokens=45
    )
    if "Error" in ack_resp_text or "OpenAI client not available" in ack_resp_text:
        fallbacks = ["Okay, thank you for that.", "Understood, thanks.", "That's clear, thank you.", "Appreciate the detail."]
        return fallbacks[hash(answer_text) % len(fallbacks)]
    ack_reply = ack_resp_text.strip()
    if not ack_reply:
        return "Understood."
    if ack_reply.endswith('?'):
        ack_reply = ack_reply[:-1] + '.'
    if not re.search(r'[.!?]$', ack_reply):
        ack_reply += '.'
    if '?' in ack_reply:
        ack_reply = ack_reply.replace('?', '.')
    return ack_reply

def authenticate_user_db_old(username_auth, password_auth):
    try:
        with sqlite3.connect('users.db') as conn_auth:
            cursor_auth = conn_auth.cursor()
            cursor_auth.execute('SELECT Allowed FROM users WHERE Username = ? AND Password = ?', (username_auth, password_auth))
            result_auth = cursor_auth.fetchone()
        return result_auth[0] if result_auth else None
    except sqlite3.Error as e_auth_db:
        logging.error(f"Authentication DB error for user '{username_auth}': {e_auth_db}", exc_info=True); return None
    except Exception as e_auth_generic:
        logging.error(f"Generic authentication error for user '{username_auth}': {e_auth_generic}", exc_info=True); return None

def analyze_frame_for_visuals(cv_frame):
    try:
        gray_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
        cascade_path_cv = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path_cv):
            logging.error(f"Visual Analysis: Haar cascade file not found at {cascade_path_cv}")
            return {'eye_contact': False, 'confidence': 1.0, 'emotion': 'unknown', 'timestamp': time.time(), 'error': 'Cascade file missing'}
        face_cascade_cv = cv2.CascadeClassifier(cascade_path_cv)
        if face_cascade_cv.empty():
            logging.error("Visual Analysis: Failed to load Haar cascade classifier.")
            return {'eye_contact': False, 'confidence': 1.0, 'emotion': 'unknown', 'timestamp': time.time(), 'error': 'Classifier load failed'}
        faces_detected = face_cascade_cv.detectMultiScale(gray_frame, scaleFactor=1.12, minNeighbors=6, minSize=(120, 120))
        eye_contact_flag = len(faces_detected) > 0
        confidence_val = 2.0
        if eye_contact_flag:
            confidence_val = 5.0 + min(3.0, len(faces_detected) * 0.4)
        else:
            logging.warning("Visual Analysis: No faces detected in frame.")
            confidence_val = 1.0  # Lower confidence if no face detected
        brightness_val = np.mean(gray_frame)
        emotion_label_cv = 'neutral'
        if eye_contact_flag:
            if brightness_val > 145: 
                emotion_label_cv = 'positive_leaning'
            elif brightness_val < 65: 
                emotion_label_cv = 'negative_leaning'
        return {
            'eye_contact': eye_contact_flag,
            'confidence': round(confidence_val, 1),
            'emotion': emotion_label_cv,
            'timestamp': time.time(),
            'error': None
        }
    except Exception as e_analyze:
        logging.error(f"Visual Analysis: Error in analyze_frame_for_visuals: {e_analyze}", exc_info=True)
        return {
            'eye_contact': False,
            'confidence': 1.0,
            'emotion': 'unknown',
            'timestamp': time.time(),
            'error': str(e_analyze)
        }

def capture_and_analyze_visuals_thread_func():
    global visual_analyses, visual_analysis_thread, interview_context
    cap_visual = None
    logging.info("Visual Analysis Thread: Started.")
    try:
        # Try multiple camera indices
        for index in range(3):
            logging.info(f"Visual Analysis Thread: Attempting to open camera at index {index}")
            cap_visual = cv2.VideoCapture(index)
            if cap_visual.isOpened():
                logging.info(f"Visual Analysis Thread: Camera opened successfully at index {index}")
                break
        else:
            logging.error("Visual Analysis Thread: Failed to open any webcam device after trying indices 0-2.")
            if platform.system() == "Linux":
                logging.info("Visual Analysis Thread: Running on Linux. Check camera permissions (e.g., /dev/video0) and ensure v4l2loopback or similar is configured.")
            return

        last_snapshot_taken_time = 0
        snapshot_capture_interval = 30
        while True:
            current_context_active = interview_context
            use_camera_in_context = current_context_active.get('use_camera_feature', False) if current_context_active else False
            if not use_camera_in_context or visual_analysis_thread != threading.current_thread():
                logging.info(f"Visual Analysis Thread: Stopping. use_camera_in_context: {use_camera_in_context}, thread_match: {visual_analysis_thread == threading.current_thread()}.")
                break

            ret_frame, cv_frame_cap = cap_visual.read()
            if not ret_frame or cv_frame_cap is None:
                logging.warning("Visual Analysis Thread: Failed to capture frame.")
                visual_analyses.append({
                    'eye_contact': False,
                    'confidence': 1.0,
                    'emotion': 'unknown',
                    'timestamp': time.time(),
                    'error': 'Failed to capture frame'
                })
                time.sleep(0.25)
                continue
            analysis_data = analyze_frame_for_visuals(cv_frame_cap)
            visual_analyses.append(analysis_data)
            current_ts = time.time()
            if current_context_active.get('use_camera_feature', False) and (current_ts - last_snapshot_taken_time >= snapshot_capture_interval):
                dt_str_snap = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_filename_va = f"va_snapshot_{dt_str_snap}.jpg"
                snap_filepath_va = os.path.join('uploads', 'snapshots', snap_filename_va)
                try:
                    cv2.imwrite(snap_filepath_va, cv_frame_cap)
                    logging.info(f"Visual Analysis Thread: Snapshot saved: {snap_filepath_va}")
                    last_snapshot_taken_time = current_ts
                except Exception as e_snap_va:
                    logging.error(f"Visual Analysis Thread: Failed to save snapshot: {e_snap_va}")
            time.sleep(0.3)
    except Exception as e_thread_va:
        logging.error(f"Visual Analysis Thread: Exception in main loop: {e_thread_va}", exc_info=True)
    finally:
        if cap_visual:
            cap_visual.release()
            logging.info("Visual Analysis Thread: Camera released.")
        logging.info("Visual Analysis Thread: Terminated.")

def calculate_visual_score():
    if not visual_analyses:
        logging.warning("Calculate Visual Score: No visual data available.")
        return 0.0, "No visual data was captured for scoring."
    try:
        num_samples_va = len(visual_analyses)
        valid_samples = [item for item in visual_analyses if item.get('error') is None]
        if not valid_samples:
            logging.warning("Calculate Visual Score: No valid visual samples.")
            return 0.0, "No valid visual data was captured for scoring."
        ec_frames = sum(1 for item in valid_samples if item.get('eye_contact', False))
        avg_conf_va = sum(item.get('confidence', 1.0) for item in valid_samples) / len(valid_samples)
        pos_emo_frames = sum(1 for item in valid_samples if 'positive' in item.get('emotion', ''))
        ec_ratio_va = ec_frames / len(valid_samples) if valid_samples else 0.0
        conf_norm_va = min(max(avg_conf_va, 0), 10) / 10.0
        pos_emo_ratio_va = pos_emo_frames / len(valid_samples) if valid_samples else 0.0
        w_ec, w_conf, w_emo = 0.45, 0.35, 0.20
        score_ec_comp = ec_ratio_va * 10
        score_conf_comp = conf_norm_va * 10
        score_emo_comp = pos_emo_ratio_va * 10
        final_visual_score_val = (score_ec_comp * w_ec) + (score_conf_comp * w_conf) + (score_emo_comp * w_emo)
        feedback_text_va = (
            f"Estimated eye contact maintained approximately {round(ec_ratio_va*100)}% of samples. "
            f"Average perceived confidence: {round(avg_conf_va,1)}/10. "
            f"Positive emotion cues observed in approximately {round(pos_emo_ratio_va*100)}% of samples."
        )
        return round(final_visual_score_val, 1), feedback_text_va
    except ZeroDivisionError:
        logging.warning("Calculate Visual Score: Division by zero.")
        return 0.0, "Not enough valid visual data."
    except Exception as e_cvs:
        logging.error(f"Calculate Visual Score: Error: {e_cvs}", exc_info=True)
        return 0.0, "Error calculating visual score."

@app.route('/')
def index_route():
    if 'allowed_user_type' not in session: return redirect(url_for('login_html_route'))
    return render_template('index.html')

@app.route('/login.html')
def login_html_route(): return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post_route():
    try:
        username_form = request.form.get('username'); password_form = request.form.get('password')
        if not username_form or not password_form: return jsonify({'success': False, 'error': 'Username and password are required.'}), 400
        allowed_user_type_from_db = authenticate_user_db_old(username_form, password_form)
        if allowed_user_type_from_db:
            session['allowed_user_type'] = allowed_user_type_from_db; session['username'] = username_form
            logging.info(f"User '{username_form}' logged in as type '{allowed_user_type_from_db}'.")
            return jsonify({'success': True, 'allowed': allowed_user_type_from_db})
        else:
            logging.warning(f"Failed login attempt for username: '{username_form}'.")
            return jsonify({'success': False, 'error': 'Invalid username or password.'}), 401
    except Exception as e_login:
        logging.error(f"Login Error for user '{request.form.get('username', 'N/A')}': {e_login}", exc_info=True)
        return jsonify({'success': False, 'error': 'An internal server error occurred during login.'}), 500

@app.route('/logout')
def logout_route():
    global visual_analysis_thread, visual_analyses, interview_context, qna_evaluations
    username_logout = session.get('username', 'User')
    logging.info(f"Logout initiated for user {username_logout}.")
    try:
        current_ic_ref = interview_context
        if current_ic_ref and current_ic_ref.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
            logging.info(f"Logout: Signaling visual analysis thread to stop for {username_logout}.")
            current_ic_ref['use_camera_feature'] = False
            visual_analysis_thread.join(timeout=1.5)
            if visual_analysis_thread.is_alive():
                 logging.warning(f"Logout: Visual analysis thread for {username_logout} did not terminate gracefully.")
            visual_analysis_thread = None

        session.clear()
        visual_analyses = []
        qna_evaluations = []
        interview_context = {}
        logging.info(f"User {username_logout} logged out. Session and global states have been reset.")
        return redirect(url_for('login_html_route'))
    except Exception as e_logout:
        logging.error(f"Error during logout for {username_logout}: {e_logout}", exc_info=True)
        session.clear()
        visual_analysis_thread = None; visual_analyses = []; interview_context = {}; qna_evaluations = []
        return redirect(url_for('login_html_route'))

@app.route('/capture_snapshot', methods=['POST'])
def capture_snapshot_route():
    try:
        if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized"}), 401
        data = request.get_json()
        image_data_url_snap = data.get('image_data_url')
        if not image_data_url_snap: return jsonify({"error": "No image data (image_data_url) received for snapshot."}), 400
        try:
            img_header, img_encoded_data = image_data_url_snap.split(",", 1); img_bytes = base64.b64decode(img_encoded_data)
            snap_ts = datetime.now().strftime("%Y%m%d_%H%M%S_frontend_snap"); snap_fname_fe = f"fe_snapshot_{snap_ts}.jpg"
            snap_fpath_fe = os.path.join('uploads', 'snapshots', snap_fname_fe)
            with open(snap_fpath_fe, "wb") as f_snap: f_snap.write(img_bytes)
            conn = sqlite3.connect('interview_data.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO snapshots (username, timestamp, image_path)
                VALUES (?, ?, ?)
            ''', (
                session.get('username', 'anonymous'),
                datetime.now().isoformat(),
                snap_fpath_fe
            ))
            conn.commit()
            conn.close()
            logging.info(f"Frontend snapshot saved successfully: {snap_fpath_fe}")
            return jsonify({"message": f"Snapshot captured from frontend and saved as {snap_fname_fe}."}), 200
        except ValueError: return jsonify({"error": "Invalid image data URL format for snapshot."}), 400
        except Exception as e_save_snap:
            logging.error(f"Error processing/saving frontend snapshot: {e_save_snap}", exc_info=True)
            return jsonify({"error": "Failed to process or save the snapshot data."}), 500
    except Exception as e_snap_route:
        logging.error(f"Error in /capture_snapshot route: {e_snap_route}", exc_info=True)
        return jsonify({"error": "Server error handling snapshot."}), 500

@app.route('/start_interview', methods=['POST'])
def start_interview_route():
    global qna_evaluations, current_use_voice_mode, interview_context, listening_active, visual_analysis_thread, visual_analyses
    try:
        if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized. Please log in again."}), 401
        
        qna_evaluations = []
        visual_analyses = []
        interview_context = interview_context_template.copy()
        interview_context['questions_already_asked'] = set()
        interview_context['generated_resume_questions_cache'] = []

        if visual_analysis_thread and visual_analysis_thread.is_alive():
            logging.warning("Start Interview: Previous visual analysis thread was active. Signaling it to stop.")
            old_thread = visual_analysis_thread
            visual_analysis_thread = None
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=0.5)
                if old_thread.is_alive():
                    logging.warning("Start Interview: Old visual thread did not stop quickly.")
        visual_analysis_thread = None

        allowed_user_type_sess = session['allowed_user_type']
        current_use_voice_mode = request.form.get('mode') == 'voice'
        track_form = request.form.get('interview_track'); sub_track_form = request.form.get('sub_track', '')
        use_camera_form = request.form.get('use_camera') == 'true'

        interview_context.update({
            'current_interview_track': track_form, 'current_sub_track': sub_track_form,
            'use_camera_feature': use_camera_form,
            'current_job_description': f"{allowed_user_type_sess} Candidate for {track_form} track"
        })
        job_key_map = 'mba' if allowed_user_type_sess == 'MBA' else 'bank'

        resume_file_form = request.files.get('resume')
        if not resume_file_form: return jsonify({"error": "Resume file is mandatory."}), 400
        resume_text_content = ""
        temp_resume_path_start = os.path.join('uploads', f"temp_resume_{session.get('username','default')}_{resume_file_form.filename}")
        try:
            resume_file_form.save(temp_resume_path_start)
            if temp_resume_path_start.lower().endswith('.pdf'):
                with pdfplumber.open(temp_resume_path_start) as pdf_doc: resume_text_content = ''.join(p.extract_text() or '' for p in pdf_doc.pages if p.extract_text())
            elif temp_resume_path_start.lower().endswith('.docx'): resume_text_content = docx2txt.process(temp_resume_path_start)
            else: return jsonify({"error": "Unsupported resume file type (PDF or DOCX only)."}), 400
            if not resume_text_content.strip(): resume_text_content = "Resume content seems empty or could not be extracted."
        except Exception as e_res_proc:
            logging.error(f"Error processing resume '{resume_file_form.filename}': {e_res_proc}", exc_info=True); return jsonify({"error": f"Could not process resume: {str(e_res_proc)}"}), 500
        finally:
            if os.path.exists(temp_resume_path_start):
                try: os.remove(temp_resume_path_start)
                except OSError as e_del_res: logging.warning(f"Could not delete temp resume file '{temp_resume_path_start}': {e_del_res}")

        if not interview_context['generated_resume_questions_cache']:
             interview_context['generated_resume_questions_cache'] = generate_resume_questions(resume_text_content, job_key_map, interview_context['questions_already_asked'])
        for q_res_gen in interview_context['generated_resume_questions_cache']:
            interview_context['questions_already_asked'].add(normalize_text(q_res_gen))

        current_q_list_intermediate = []
        job_specific_pdf_structure = structure.get(job_key_map, {})

        if job_key_map == 'mba':
            if track_form == "resume":
                predef_qs = [q_obj['text'] for q_obj in job_specific_pdf_structure.get('resume_flow', [])[:3]]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'])
                for q_pd in predef_qs:
                    if normalize_text(q_pd) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_pd)
            elif track_form == "school_based":
                school_data = job_specific_pdf_structure.get('school_based', defaultdict(list))
                school_qs_track = [q_obj['text'] for q_obj in school_data.get(sub_track_form, [])]
                if not school_qs_track: school_qs_track = [q_obj['text'] for sub_list in school_data.values() for q_obj in sub_list]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'][:5])
                for q_school in school_qs_track:
                    if normalize_text(q_school) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_school)
            elif track_form == "interest_areas":
                interest_data = job_specific_pdf_structure.get('interest_areas', defaultdict(list))
                interest_qs_track = [q_obj['text'] for q_obj in interest_data.get(sub_track_form, [])]
                if not interest_qs_track: interest_qs_track = [q_obj['text'] for sub_list in interest_data.values() for q_obj in sub_list]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'][:5])
                for q_interest in interest_qs_track:
                     if normalize_text(q_interest) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_interest)
        elif job_key_map == 'bank':
            if track_form == "resume":
                predef_qs_bank = [q_obj['text'] for q_obj in job_specific_pdf_structure.get('resume_flow', [])[:3]]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'])
                for q_pd_bank in predef_qs_bank:
                    if normalize_text(q_pd_bank) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_pd_bank)
            elif track_form == "bank_type":
                bank_type_data = job_specific_pdf_structure.get('bank_type', defaultdict(list))
                bank_qs_track = [q_obj['text'] for q_obj in bank_type_data.get(sub_track_form, [])]
                if not bank_qs_track: bank_qs_track = [q_obj['text'] for sub_list in bank_type_data.values() for q_obj in sub_list]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'][:5])
                for q_bank_type in bank_qs_track:
                    if normalize_text(q_bank_type) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_bank_type)
            elif track_form == "technical_analytical":
                tech_ana_data = job_specific_pdf_structure.get('technical_analytical', defaultdict(list))
                tech_qs_track = [q_obj['text'] for q_obj in tech_ana_data.get(sub_track_form, [])]
                if not tech_qs_track: tech_qs_track = [q_obj['text'] for sub_list in tech_ana_data.values() for q_obj in sub_list]
                current_q_list_intermediate = list(interview_context['generated_resume_questions_cache'][:5])
                for q_tech in tech_qs_track:
                    if normalize_text(q_tech) not in interview_context['questions_already_asked']: current_q_list_intermediate.append(q_tech)

        final_interview_questions_for_session = []
        temp_asked_this_specific_list_build = set()
        for q_text_final_candidate in current_q_list_intermediate:
            stripped_q_final = strip_numbering(q_text_final_candidate)
            norm_stripped_q_final = normalize_text(stripped_q_final)
            if norm_stripped_q_final not in interview_context['questions_already_asked'] and \
               norm_stripped_q_final not in temp_asked_this_specific_list_build:
                final_interview_questions_for_session.append(stripped_q_final)
                temp_asked_this_specific_list_build.add(norm_stripped_q_final)

        interview_context['questions_list'] = final_interview_questions_for_session
        for q_final_sess in final_interview_questions_for_session:
            interview_context['questions_already_asked'].add(normalize_text(q_final_sess))

        if not interview_context['questions_list']:
            logging.warning(f"User's original logic yielded no questions for '{track_form}/{sub_track_form}'. Using hardcoded fallbacks.")
            interview_context['questions_list'] = ["Please describe your most relevant experience.", "What are your key strengths for this role/program?"]
            for q_fb_final in interview_context['questions_list']: interview_context['questions_already_asked'].add(normalize_text(q_fb_final))

        interview_context['icebreaker_was_prepended'] = False
        interview_context['prepended_icebreaker_text'] = None
        if interview_context['use_camera_feature']:
            logging.info("Camera ON. Attempting to generate formal icebreaker.")
            initial_frame_b64_ice = capture_initial_frame_data_for_question()
            if initial_frame_b64_ice:
                icebreaker_question_text = generate_environment_icebreaker_question(initial_frame_b64_ice)
                if icebreaker_question_text:
                    norm_icebreaker_text = normalize_text(icebreaker_question_text)
                    if norm_icebreaker_text not in interview_context['questions_already_asked']:
                        interview_context['questions_list'].insert(0, icebreaker_question_text)
                        interview_context['questions_already_asked'].add(norm_icebreaker_text)
                        interview_context['icebreaker_was_prepended'] = True
                        interview_context['prepended_icebreaker_text'] = icebreaker_question_text
                        logging.info(f"Icebreaker prepended: '{icebreaker_question_text}'")
                    else: logging.info("Generated icebreaker was duplicate, not adding.")
                else: logging.info("Failed to generate suitable icebreaker from visual data.")
            else: logging.info("Failed to capture frame for icebreaker.")

        if not interview_context['questions_list']:
            logging.error("FATAL: No questions available for interview."); return jsonify({"error": "System could not prepare any questions."}), 500

        interview_context['current_q_idx'] = 0
        listening_active = True

        if interview_context['use_camera_feature']:
            visual_analyses = []
            visual_analysis_thread = threading.Thread(target=capture_and_analyze_visuals_thread_func, daemon=True)
            visual_analysis_thread.start()
            logging.info("Visual analysis background thread started.")

        logging.info(f"Interview starting for {allowed_user_type_sess}, track '{track_form}'. Total questions in list: {len(interview_context['questions_list'])}")
        return jsonify({
            "message": f"Starting {allowed_user_type_sess} interview. Focus: {track_form}.",
            "total_questions": len(interview_context['questions_list']),
            "current_question": interview_context['questions_list'][0],
            "question_number": 1,
            "use_voice": current_use_voice_mode,
            "use_camera": interview_context['use_camera_feature'],
            "listening_active": listening_active if current_use_voice_mode else False
        })
    except Exception as e_start_interview:
        logging.error(f"Critical error in /start_interview route: {e_start_interview}", exc_info=True)
        return jsonify({"error": f"A major server error occurred during interview setup: {str(e_start_interview)}"}), 500

def calculate_final_overall_score(current_qna_evaluations, visual_score_0_to_10_val=None):
    try:
        qna_max_score_contribution = 90.0; visual_max_score_contribution = 10.0
        actual_qna_score_total = sum(item.get("score", 0) for item in current_qna_evaluations if isinstance(item.get("score"), (int, float)))
        possible_qna_score_total = len(current_qna_evaluations) * 10
        if not current_qna_evaluations or possible_qna_score_total == 0:
            qna_percentage_achieved = 0.0
        else:
            qna_percentage_achieved = actual_qna_score_total / possible_qna_score_total
        qna_weighted_contribution = qna_percentage_achieved * qna_max_score_contribution
        visual_weighted_contribution = 0.0
        if isinstance(visual_score_0_to_10_val, (int, float)) and visual_score_0_to_10_val is not None:
            visual_percentage_achieved = visual_score_0_to_10_val / 10.0
            visual_weighted_contribution = visual_percentage_achieved * visual_max_score_contribution
        final_overall_score_calculated = qna_weighted_contribution + visual_weighted_contribution
        return round(max(0.0, min(100.0, final_overall_score_calculated)), 2)
    except Exception as e_calc_score:
        logging.error(f"Error calculating final overall score: {e_calc_score}", exc_info=True); return 0.0

@app.route('/submit_answer', methods=['POST'])
def submit_answer_route():
    global qna_evaluations, current_use_voice_mode, interview_context, listening_active, visual_analysis_thread
    try:
        if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized. Session may have expired."}), 401
        if not interview_context or 'questions_list' not in interview_context or \
           not isinstance(interview_context.get('questions_list'), list) or \
           'current_q_idx' not in interview_context:
            logging.error("Submit Answer: Interview context corrupted or not initialized.")
            calculated_final_visual_score, visual_feedback_on_error = (0.0, "N/A (Session error)")
            if interview_context and interview_context.get('use_camera_feature', False):
                 calculated_final_visual_score, visual_feedback_on_error = calculate_visual_score()
            overall_score_on_error = calculate_final_overall_score(qna_evaluations, calculated_final_visual_score)
            if interview_context and interview_context.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
                interview_context['use_camera_feature'] = False; visual_analysis_thread.join(timeout=0.7); visual_analysis_thread = None
            return jsonify({"reply": "Critical error with session. Interview ending.", "finished": True, "evaluations": qna_evaluations, "overall_score": overall_score_on_error, "visual_score_details": {"score": calculated_final_visual_score, "feedback": visual_feedback_on_error}, "status": "Error: Session Failure"}), 500

        if not request.is_json: return jsonify({"error": "Invalid request: JSON expected."}), 400
        data_payload = request.get_json()
        answer_text_from_user = data_payload.get('answer', "").strip()

        stop_interview_phrases = ["stop this interview", "end this interview", "stop the interview", "end the interview"]
        normalized_answer_for_check = answer_text_from_user.lower()
        user_wants_to_stop = any(stop_phrase in normalized_answer_for_check for stop_phrase in stop_interview_phrases)

        if user_wants_to_stop:
            user_name_log = session.get('username', 'N/A_User')
            logging.info(f"User '{user_name_log}' requested to stop/end interview. Answer: '{answer_text_from_user}'.")
            calculated_final_visual_score, visual_feedback_on_stop = (0.0, "N/A (Interview stopped by user)")
            if interview_context.get('use_camera_feature', False): calculated_final_visual_score, visual_feedback_on_stop = calculate_visual_score()
            overall_score_on_stop = calculate_final_overall_score(qna_evaluations, calculated_final_visual_score)
            job_description_for_feedback_gen = interview_context.get("current_job_description", f"{session.get('allowed_user_type', 'Candidate')} Profile")
            for eval_item_on_stop in qna_evaluations:
                if not eval_item_on_stop.get('feedback'): eval_item_on_stop['feedback'] = generate_answer_feedback(eval_item_on_stop.get('question', 'Unknown'), eval_item_on_stop.get('answer', ''), job_description_for_feedback_gen)

            listening_active = False
            if interview_context.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
                interview_context['use_camera_feature'] = False; visual_analysis_thread.join(timeout=0.7); visual_analysis_thread = None
            return jsonify({"reply": "Interview stopped as per your request.", "finished": True, "evaluations": qna_evaluations, "overall_score": overall_score_on_stop, "visual_score_details": {"score": calculated_final_visual_score, "feedback": visual_feedback_on_stop}, "status": "Disqualified: User Request"})

        current_question_idx_val = interview_context.get('current_q_idx', -1)
        if not (0 <= current_question_idx_val < len(interview_context['questions_list'])):
            logging.error(f"Submit Answer: Invalid current_q_idx ({current_question_idx_val}). List len ({len(interview_context.get('questions_list',[]))}). Ending.")
            vis_score_idx_err, vis_feed_idx_err = (0.0, "N/A (Q index error)")
            if interview_context.get('use_camera_feature'): vis_score_idx_err, vis_feed_idx_err = calculate_visual_score()
            overall_score_idx_err = calculate_final_overall_score(qna_evaluations, vis_score_idx_err)
            if interview_context.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
                interview_context['use_camera_feature'] = False; visual_analysis_thread.join(timeout=0.7); visual_analysis_thread = None
            return jsonify({"reply": "Issue with question sequence. Interview concluding.", "finished": True, "evaluations": qna_evaluations, "overall_score": overall_score_idx_err, "visual_score_details": {"score": vis_score_idx_err, "feedback": vis_feed_idx_err}, "status": "Error: Q Index Problem"})

        question_text_being_answered = interview_context['questions_list'][current_question_idx_val]
        answer_text_to_process = answer_text_from_user if answer_text_from_user else "No specific answer was provided."

        is_current_question_the_icebreaker = False
        if interview_context.get('icebreaker_was_prepended') and \
           current_question_idx_val == 0 and \
           question_text_being_answered == interview_context.get('prepended_icebreaker_text'):
            is_current_question_the_icebreaker = True

        job_key_for_ai = 'mba' if session.get('allowed_user_type') == 'MBA' else 'bank'
        job_desc_for_ai = interview_context.get("current_job_description", f"{session.get('allowed_user_type', 'Candidate')} Profile")
        conversational_ack_reply = generate_conversational_reply(answer_text_to_process, job_key_for_ai)
        ai_detailed_eval_str, ai_weighted_score_val = evaluate_response_with_ai_scoring(question_text_being_answered, answer_text_to_process, job_desc_for_ai)
        user_summary_feedback_str = generate_answer_feedback(question_text_being_answered, answer_text_to_process, job_desc_for_ai)

        qna_evaluations.append({"question": question_text_being_answered, "answer": answer_text_to_process, "evaluation": ai_detailed_eval_str, "score": ai_weighted_score_val, "feedback": user_summary_feedback_str})
        interview_context["previous_answers_list"].append(answer_text_to_process)
        interview_context["scores_list"].append(ai_weighted_score_val)
        interview_context['questions_already_asked'].add(normalize_text(question_text_being_answered))

        if is_current_question_the_icebreaker:
            logging.info("Answer to icebreaker received. Skipping follow-up for it. Resetting depth counter.")
            interview_context["question_depth_counter"] = 0
        else:
            current_depth = interview_context.get("question_depth_counter", 0)
            max_depth = interview_context.get("max_followup_depth", 2)
            if current_depth < max_depth:
                follow_up_q_generated_text = generate_next_question(question_text_being_answered, answer_text_to_process, ai_weighted_score_val, interview_context.get("current_interview_track", "unknown"), job_key_for_ai, interview_context.get('questions_already_asked', set()))
                if follow_up_q_generated_text:
                    interview_context['questions_list'].insert(current_question_idx_val + 1, follow_up_q_generated_text)
                    interview_context['questions_already_asked'].add(normalize_text(follow_up_q_generated_text))
                    interview_context["question_depth_counter"] = current_depth + 1
                    logging.info(f"Follow-up inserted: '{follow_up_q_generated_text}'. Depth: {interview_context['question_depth_counter']}")
                else: interview_context["question_depth_counter"] = 0
            else: interview_context["question_depth_counter"] = 0

        interview_context['current_q_idx'] += 1

        if interview_context['current_q_idx'] < len(interview_context['questions_list']):
            next_question_to_ask_text = interview_context['questions_list'][interview_context['current_q_idx']]
            listening_active = True
            return jsonify({"reply": conversational_ack_reply, "current_question": next_question_to_ask_text, "question_number": interview_context['current_q_idx'] + 1, "total_questions": len(interview_context['questions_list']), "next_question": True, "listening_active": listening_active if current_use_voice_mode else False})
        else:
            logging.info("All questions asked. Interview concluding normally.")
            final_visual_score_val_norm, visual_feedback_text_norm = (0.0, "N/A (Camera not used/error)")
            if interview_context.get('use_camera_feature', False): final_visual_score_val_norm, visual_feedback_text_norm = calculate_visual_score()
            overall_score_val_norm = calculate_final_overall_score(qna_evaluations, final_visual_score_val_norm)
            for eval_item_norm in qna_evaluations:
                if not eval_item_norm.get('feedback'): eval_item_norm['feedback'] = generate_answer_feedback(eval_item_norm.get('question', 'Unknown'), eval_item_norm.get('answer', 'N/A'), job_desc_for_ai)

            listening_active = False
            if interview_context.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
                interview_context['use_camera_feature'] = False; visual_analysis_thread.join(timeout=0.7); visual_analysis_thread = None
            return jsonify({"reply": "Thank you for completing the interview.", "finished": True, "evaluations": qna_evaluations, "overall_score": overall_score_val_norm, "visual_score_details": {"score": final_visual_score_val_norm, "feedback": visual_feedback_text_norm}, "status": "Completed Successfully"})
    except Exception as e_submit_ans:
        logging.error(f"Critical error in /submit_answer: {e_submit_ans}", exc_info=True)
        vis_score_exc, vis_feed_exc = (0.0, "N/A (Exception during submit)")
        current_ic = interview_context
        if current_ic and current_ic.get('use_camera_feature'):
            vis_score_exc, vis_feed_exc = calculate_visual_score()
        overall_score_exc = calculate_final_overall_score(qna_evaluations, vis_score_exc)
        if current_ic and current_ic.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
            current_ic['use_camera_feature'] = False; visual_analysis_thread.join(timeout=0.7); visual_analysis_thread = None

        return jsonify({"error": f"Critical server error: {str(e_submit_ans)}.", "reply": "Unexpected problem processing answer.", "finished": True, "evaluations": qna_evaluations, "overall_score": overall_score_exc, "visual_score_details": {"score": vis_score_exc, "feedback": vis_feed_exc}, "status": "Error: Unhandled Exception"}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        question = data.get('question')
        feedback = data.get('feedback')

        if not question or not feedback:
            return jsonify({'success': False, 'error': 'Incomplete data received.'}), 400

        feedback_entry = f"{datetime.now()} - Question: {question}\nFeedback: {feedback}\n\n"
        with open('feedback_log.txt', 'a', encoding='utf-8') as f:
            f.write(feedback_entry)

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error saving feedback: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Server error while saving feedback.'}), 500

@app.route('/submit_bulk_feedback', methods=['POST'])
def submit_bulk_feedback():
    try:
        data = request.get_json()
        entries = data.get('entries', [])

        if not entries:
            return jsonify({'success': False, 'error': 'No feedback entries received.'}), 400

        with open('bulk_feedback_log.txt', 'a', encoding='utf-8') as f:
            for entry in entries:
                question = entry.get('question')
                feedback = entry.get('feedback')
                f.write(f"{datetime.now()} - Question: {question}\nFeedback: {feedback}\n\n")

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Bulk feedback error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal error.'}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech_route():
    try:
        if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized access."}), 401
        if not client: return jsonify({"error": "TTS service unavailable."}), 503
        if not request.is_json: return jsonify({"error": "Invalid request: JSON expected."}), 400
        data_tts = request.get_json(); text_for_speech = data_tts.get('text', ''); voice_model_selection = data_tts.get('voice', 'alloy')
        if not text_for_speech.strip(): return jsonify({"error": "Text for speech required."}), 400
        supported_openai_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'sage']
        final_voice_model = voice_model_selection if voice_model_selection in supported_openai_voices else 'alloy'
        logging.info(f"Generating TTS for text: '{text_for_speech[:50]}...' with voice: '{final_voice_model}'")
        try:
            openai_tts_response = client.audio.speech.create(
                model="tts-1", 
                voice=final_voice_model, 
                input=text_for_speech, 
                response_format="mp3",
                timeout=10  # Set a 10-second timeout
            )
            logging.info(f"TTS generated successfully for '{text_for_speech[:50]}...'")
            return Response(openai_tts_response.content, mimetype='audio/mp3')
        except TimeoutError as e_timeout:
            logging.error(f"TTS Timeout Error: Request to OpenAI timed out after 10 seconds: {e_timeout}")
            return jsonify({"error": "TTS generation timed out. Please try again or proceed without audio."}), 504
        except Exception as e_openai:
            logging.error(f"TTS OpenAI Error: {e_openai}", exc_info=True)
            error_message = f"TTS generation failed: {str(e_openai)}"
            if hasattr(e_openai, 'response') and e_openai.response:
                try:
                    err_content = e_openai.response.json()
                    error_message = f"TTS generation failed: {err_content.get('error', {}).get('message', str(e_openai))}"
                except:
                    pass
            return jsonify({"error": error_message}), 500
    except Exception as e_tts_route:
        logging.error(f"TTS Route Error: {e_tts_route}", exc_info=True)
        return jsonify({"error": f"Server error during TTS processing: {str(e_tts_route)}"}), 500

def init_db():
    conn = sqlite3.connect('interview_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            question TEXT,
            answer TEXT,
            evaluation TEXT,
            score INTEGER,
            feedback TEXT,
            timestamp TEXT
        )
    ''')

    conn.commit()
    conn.close()

@app.route('/submit_evaluations', methods=['POST'])
def submit_evaluations():
    try:
        data = request.get_json()
        evaluations = data.get('evaluations', [])

        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()

        for eval in evaluations:
            cursor.execute('''
                INSERT INTO evaluations (username, question, answer, evaluation, score, feedback, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.get('username', 'anonymous'),
                eval.get('question'),
                eval.get('answer'),
                eval.get('evaluation'),
                eval.get('score'),
                eval.get('feedback', ''),
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

        return jsonify({'success': True})

    except Exception as e:
        logging.error(f"Error saving evaluations: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001, host="0.0.0.0")
