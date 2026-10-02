"""Configuration for the Mental Health Chatbot prototype.

EDIT the placeholders marked [EDIT] before any real deployment or demonstration.
"""
import os

APP_TITLE = "Mental Health Support Chatbot"
APP_SUBTITLE = "Anonymous first-line emotional support for students"

# ---- Storage -------------------------------------------------------------
DB_PATH = os.environ.get("MH_DB_PATH", "mh_chatbot.db")
# Set MH_STORE_MESSAGES=0 to avoid storing the text of conversations (privacy mode).
STORE_MESSAGE_TEXT = os.environ.get("MH_STORE_MESSAGES", "1") != "0"

# ---- NLP -----------------------------------------------------------------
# auto = use the Hugging Face model if it can be loaded, otherwise the built-in lexicon
# transformer = same as auto (fallback still protects the app from crashing)
# lexicon = never load a model (fast, works fully offline)
EMOTION_BACKEND = os.environ.get("EMOTION_BACKEND", "auto")
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")

# ---- Counsellor dashboard --------------------------------------------------
# Set COUNSELLOR_PASSWORD in the environment. If it is empty, a weak development
# password ("counsellor") is used and a warning is shown on the dashboard.
COUNSELLOR_PASSWORD = os.environ.get("COUNSELLOR_PASSWORD", "")

# ---- Crisis resources -------------------------------------------------------
# 112 is Nigeria's national emergency number. Add verified local numbers below. [EDIT]
CRISIS_RESOURCES = [
    {"name": "National emergency number (Nigeria)", "contact": "112"},
    {"name": "University counselling unit", "contact": "[EDIT: add counselling unit phone / office location]"},
    {"name": "University health centre (24-hour)", "contact": "[EDIT: add health centre phone / location]"},
    {"name": "Trusted crisis helpline", "contact": "[EDIT: add a verified Nigerian mental health helpline number]"},
]

# ---- Counselling directory used by the "Therapist referral" pathway ---------
# focus tags: depression, anxiety, general, crisis. [EDIT] with real services.
COUNSELLING_DIRECTORY = [
    {"id": "campus-counselling", "name": "University Counselling Unit", "type": "Institutional counsellor",
     "focus": ["general", "depression", "anxiety"], "availability": "[EDIT: e.g. Mon-Fri, 9am-4pm]",
     "contact": "[EDIT: phone / office location]"},
    {"id": "clinical-psychologist", "name": "Clinical Psychologist (Student Health Services)", "type": "Clinical psychologist",
     "focus": ["depression", "anxiety"], "availability": "[EDIT: by appointment]",
     "contact": "[EDIT: phone / email]"},
    {"id": "psychiatric-service", "name": "Hospital Psychiatric / Mental Health Service", "type": "Psychiatric service",
     "focus": ["crisis", "depression", "anxiety"], "availability": "[EDIT: e.g. 24-hour emergency unit]",
     "contact": "[EDIT: phone / location]"},
    {"id": "peer-support", "name": "Student Peer Support Group", "type": "Peer support",
     "focus": ["general"], "availability": "[EDIT: weekly meeting time]",
     "contact": "[EDIT: contact person]"},
]

DISCLAIMER = ("This chatbot offers supportive conversation only. It is **not** a doctor or therapist, "
              "it does **not** diagnose any condition, and it is not a substitute for professional care. "
              "If you are in danger, call **112** or go to the nearest hospital.")
