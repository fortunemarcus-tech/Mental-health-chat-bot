"""Core logic for the Mental Health Chatbot prototype (no Streamlit dependency).

Modules mapped to the Tier 3 architecture:
    Chat interface      -> app.py
    Question engine     -> ScreeningSession
    Scoring module      -> score_phq9 / score_gad7 / severity bands
    Risk classifier     -> classify_risk
    Secure database     -> Database (SQLite)
    Output pathways     -> build_result_text, match_resources, Database.create_referral,
                           crisis_message + Database.create_alert
NLP layer: EmotionDetector (Hugging Face transformer with lexicon fallback),
           classify_intent, is_crisis, Responder.
"""
import json
import os
import random
import re
import sqlite3
import urllib.request
import uuid
from datetime import datetime, timezone

import config


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# SECURE DATABASE
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    session_id TEXT PRIMARY KEY, start_time TEXT NOT NULL, end_time TEXT);
CREATE TABLE IF NOT EXISTS messages(
    message_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, timestamp TEXT NOT NULL, detected_emotion TEXT);
CREATE TABLE IF NOT EXISTS mood_entries(
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, mood_label TEXT NOT NULL,
    mood_value INTEGER NOT NULL, timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS screening_results(
    result_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, phq9_score INTEGER NOT NULL,
    gad7_score INTEGER NOT NULL, risk_level TEXT NOT NULL, timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts(
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, source TEXT NOT NULL,
    snippet TEXT, status TEXT NOT NULL DEFAULT 'open', timestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS referrals(
    referral_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, resource_id TEXT NOT NULL,
    preferred_time TEXT, contact TEXT, status TEXT NOT NULL DEFAULT 'requested', timestamp TEXT NOT NULL);
"""

MOOD_LABELS = {1: "Very low", 2: "Low", 3: "Okay", 4: "Good", 5: "Very good"}


class Database:
    def __init__(self, path=None):
        self.path = path or config.DB_PATH
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _run(self, sql, params=(), fetch=False):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if fetch else None
            conn.commit()
            return rows if fetch else cur.lastrowid
        finally:
            conn.close()

    # --- sessions
    def create_session(self):
        sid = "S-" + uuid.uuid4().hex[:10]
        self._run("INSERT INTO sessions(session_id,start_time) VALUES(?,?)", (sid, now()))
        return sid

    def end_session(self, sid):
        self._run("UPDATE sessions SET end_time=? WHERE session_id=?", (now(), sid))

    # --- messages
    def log_message(self, sid, role, content, emotion=None):
        text = content if config.STORE_MESSAGE_TEXT else "[message text not stored]"
        return self._run(
            "INSERT INTO messages(session_id,role,content,timestamp,detected_emotion) VALUES(?,?,?,?,?)",
            (sid, role, text, now(), emotion))

    def get_emotion_counts(self, sid):
        rows = self._run("SELECT detected_emotion e, COUNT(*) n FROM messages WHERE session_id=? "
                         "AND role='user' AND detected_emotion IS NOT NULL GROUP BY detected_emotion", (sid,), True)
        return {r["e"]: r["n"] for r in rows}

    # --- mood
    def log_mood(self, sid, value):
        value = int(value)
        if value not in MOOD_LABELS:
            raise ValueError("mood value must be 1-5")
        return self._run("INSERT INTO mood_entries(session_id,mood_label,mood_value,timestamp) VALUES(?,?,?,?)",
                         (sid, MOOD_LABELS[value], value, now()))

    def get_mood_entries(self, sid):
        return self._run("SELECT entry_id,mood_label,mood_value,timestamp FROM mood_entries "
                         "WHERE session_id=? ORDER BY entry_id", (sid,), True)

    # --- screening
    def save_screening(self, sid, phq9, gad7, risk):
        return self._run("INSERT INTO screening_results(session_id,phq9_score,gad7_score,risk_level,timestamp) "
                         "VALUES(?,?,?,?,?)", (sid, int(phq9), int(gad7), risk, now()))

    def get_screenings(self, sid=None):
        if sid:
            return self._run("SELECT * FROM screening_results WHERE session_id=? ORDER BY result_id", (sid,), True)
        return self._run("SELECT * FROM screening_results ORDER BY result_id", (), True)

    # --- crisis alerts (human alert)
    def create_alert(self, sid, source, text=""):
        snippet = (text or "")[:200] if config.STORE_MESSAGE_TEXT else "[not stored]"
        aid = self._run("INSERT INTO alerts(session_id,source,snippet,status,timestamp) VALUES(?,?,?,?,?)",
                        (sid, source, snippet, "open", now()))
        # Optional webhook: sends NO conversation text, only identifiers.
        url = os.environ.get("ALERT_WEBHOOK_URL")
        if url:
            try:
                req = urllib.request.Request(
                    url, data=json.dumps({"alert_id": aid, "session_id": sid, "source": source,
                                          "time": now()}).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass
        return aid

    def list_alerts(self, status=None):
        if status:
            return self._run("SELECT * FROM alerts WHERE status=? ORDER BY alert_id DESC", (status,), True)
        return self._run("SELECT * FROM alerts ORDER BY alert_id DESC", (), True)

    def set_alert_status(self, alert_id, status):
        self._run("UPDATE alerts SET status=? WHERE alert_id=?", (status, int(alert_id)))

    # --- referrals
    def create_referral(self, sid, resource_id, preferred_time="", contact=""):
        return self._run("INSERT INTO referrals(session_id,resource_id,preferred_time,contact,status,timestamp) "
                         "VALUES(?,?,?,?,?,?)", (sid, resource_id, preferred_time, contact, "requested", now()))

    def list_referrals(self):
        return self._run("SELECT * FROM referrals ORDER BY referral_id DESC", (), True)


# =============================================================================
# CRISIS DETECTION  (tuned for HIGH sensitivity: false alarms are preferred to misses)
# =============================================================================
_CRISIS_PATTERNS = [
    r"\bkill(?:ing)? my ?self\b", r"\bsuicid(?:e|al)\b", r"\bend my (?:own )?life\b", r"\bend it all\b",
    r"\btake my (?:own )?life\b", r"\bdo(?:n't| not|nt) want to (?:live|be alive|exist)\b",
    r"\bwant(?:ed)? to die\b", r"\bwish (?:i|that i) (?:was|were) dead\b", r"\bbetter off dead\b",
    r"\bno reason to live\b", r"\bhurt(?:ing)? my ?self\b", r"\bself[- ]?harm\b", r"\bcut(?:ting)? my ?self\b",
    r"\bnot worth living\b", r"\bcan(?:'t|not|nt) go on\b", r"\bdisappear forever\b", r"\bend everything\b",
    # Nigerian Pidgin
    r"\bi wan die\b", r"\bi wan kill my ?self\b", r"\bi no wan live\b", r"\bmake i die\b", r"\bi go kill my ?self\b",
]
_CRISIS_RE = [re.compile(p) for p in _CRISIS_PATTERNS]


def normalize(text):
    t = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", t).strip()


def is_crisis(text):
    t = normalize(text)
    return any(p.search(t) for p in _CRISIS_RE)


def crisis_message():
    lines = "\n".join(f"- **{r['name']}:** {r['contact']}" for r in config.CRISIS_RESOURCES)
    return ("I'm really concerned about what you've shared, and I'm glad you told me. "
            "You deserve support from a real person right now.\n\n"
            f"**Please reach out now:**\n{lines}\n\n"
            "If you are in immediate danger, call **112** or go to the nearest hospital emergency unit. "
            "Please also tell someone you trust (a friend, family member, hall mate or lecturer) how you are feeling, "
            "and try not to be alone right now.\n\n"
            "A counsellor has been alerted through this system so that a human can follow up. "
            "I am not a substitute for professional care, but I'm still here with you.")


# =============================================================================
# EMOTION DETECTION
# =============================================================================
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
_LABEL_MAP = {"love": "joy", "happy": "joy", "happiness": "joy", "sad": "sadness", "afraid": "fear",
              "angry": "anger", "surprised": "surprise"}

_LEXICON = {
    "sadness": [r"sad", r"depressed", r"depression", r"hopeless", r"lonely", r"alone", r"cry(?:ing)?", r"tears",
                r"miserable", r"heartbroken", r"worthless", r"empty", r"grief", r"unhappy", r"disappointed",
                r"fail(?:ed|ure)?", r"feel(?:ing)? low", r"so low", r"no one cares", r"tired of everything",
                r"hurt(?:s|ing)?", r"i dey feel low", r"dey feel low", r"i don tire", r"tire me", r"e pain me",
                r"e dey pain me", r"i no get joy"],
    "fear": [r"scared", r"afraid", r"anxious", r"anxiety", r"worried", r"worry", r"nervous", r"panic",
             r"fear", r"terrified", r"stress(?:ed)?", r"overwhelmed", r"tension", r"i dey fear", r"i dey shake",
             r"i dey worry", r"e dey worry me"],
    "anger": [r"angry", r"mad", r"furious", r"annoyed", r"irritated", r"hate", r"rage", r"frustrated",
              r"fed up", r"upset", r"livid", r"i dey vex", r"i vex", r"vex me"],
    "joy": [r"happy", r"glad", r"excited", r"grateful", r"thankful", r"joy", r"blessed", r"proud", r"relieved",
            r"great news", r"good news", r"passed(?! away)", r"delighted", r"i dey happy", r"thank god",
            r"god don do it"],
    "surprise": [r"shocked", r"surprised", r"unexpected", r"can't believe", r"cannot believe", r"suddenly"],
    "disgust": [r"disgusted", r"disgusting", r"gross", r"sickening", r"revolting", r"nauseating"],
}
_LEX_RE = {e: [(w, re.compile(r"\b(?:%s)\b" % w),
                re.compile(r"\b(?:not|no|never|dont|don't|isnt|isn't|aint|ain't|without)\s+(?:\w+\s+)?(?:%s)\b" % w))
               for w in words] for e, words in _LEXICON.items()}


def lexicon_emotion(text):
    """Return (label, confidence) using keyword matching with simple negation handling."""
    t = normalize(text)
    scores = {e: 0 for e in _LEXICON}
    for emo, items in _LEX_RE.items():
        for _, pat, neg in items:
            if pat.search(t):
                if neg.search(t):
                    if emo == "joy":          # "not happy" -> sadness
                        scores["sadness"] += 1
                    continue                  # negated negative emotion -> ignore
                scores[emo] += 1
    best = max(scores, key=lambda e: scores[e])
    if scores[best] == 0:
        return "neutral", 0.5
    return best, min(0.95, 0.55 + 0.1 * scores[best])


class EmotionDetector:
    """Transformer-based emotion classifier with automatic lexicon fallback."""

    def __init__(self, backend="auto", model_name=None):
        self.pipe = None
        self.backend = "lexicon"
        self.error = None
        if backend in ("auto", "transformer"):
            try:
                from transformers import pipeline  # imported lazily so the app still runs without it
                self.pipe = pipeline("text-classification", model=model_name or config.EMOTION_MODEL)
                self.backend = "transformer"
            except Exception as exc:  # no internet, library missing, etc.
                self.error = str(exc)[:200]
                self.pipe = None

    def detect(self, text):
        """Return (emotion_label, confidence)."""
        lex_label, lex_conf = lexicon_emotion(text)
        if self.pipe is None:
            return lex_label, lex_conf
        try:
            res = self.pipe(text[:1000], top_k=1, truncation=True, max_length=512)
            if res and isinstance(res[0], list):
                res = res[0]
            top = res[0]
            label = str(top["label"]).lower()
            label = _LABEL_MAP.get(label, label)
            if label not in EMOTIONS:
                label = "neutral"
            score = float(top["score"])
            # Hybrid rule: when the model is unsure (e.g. Pidgin) trust a clear lexicon match.
            if (score < 0.5 or label == "neutral") and lex_label != "neutral":
                return lex_label, lex_conf
            return label, score
        except Exception:
            return lex_label, lex_conf


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================
def classify_intent(text):
    t = normalize(text)
    if re.search(r"\b(screen(?:ing)?|assess(?:ment)?|check-?in|phq|gad|questionnaire|test me|check my mental)\b", t):
        return "screening_request"
    if re.search(r"\b(therapist|counsell?or|see someone|appointment|book|referral|professional help|psychologist|psychiatrist)\b", t):
        return "referral_request"
    if re.search(r"\b(my mood|mood chart|track my mood|mood tracker)\b", t):
        return "mood_request"
    if re.fullmatch(r"(hi|hello|hey|good (morning|afternoon|evening)|how far|wetin dey|yo)\W*", t):
        return "greeting"
    if re.search(r"\b(thank(?:s| you)?|thx|i appreciate)\b", t):
        return "gratitude"
    if re.search(r"\b(bye|goodbye|good night|goodnight|see you|i am leaving|i'm leaving)\b", t):
        return "goodbye"
    if re.search(r"\b(what should i do|how can i|how do i|help me|any tips|advice|suggest|what can i do)\b", t):
        return "advice"
    return "venting"


# =============================================================================
# RESPONSE GENERATION
# =============================================================================
EMPATHY = {
    "sadness": ["I'm really sorry you're feeling this way. It sounds heavy, and it makes sense that you want to talk about it.",
                "That sounds painful. Thank you for trusting me with it.",
                "I hear you. Feeling low can be exhausting, and you don't have to go through it alone."],
    "fear": ["That sounds really stressful. Feeling anxious or afraid can be overwhelming.",
             "It's understandable to feel worried about this. Thank you for sharing it.",
             "I can hear how much this is weighing on you."],
    "anger": ["It sounds like you're really frustrated, and that's a valid feeling.",
              "I can understand why that would upset you.",
              "Thank you for telling me. Feeling angry is human, and it often points to something that matters to you."],
    "joy": ["That's lovely to hear! I'm glad something good is happening.",
            "That's wonderful. It's great to hear you in good spirits.",
            "I'm happy for you! Moments like this are worth celebrating."],
    "surprise": ["That sounds unexpected! How are you feeling about it now?",
                 "Wow, that sounds like a lot to take in."],
    "disgust": ["That sounds really unpleasant. It's okay to feel that way.",
                "I'm sorry you had to deal with that."],
    "neutral": ["Thank you for sharing that with me.", "I'm listening.", "I see. Please go on, I'm here."],
}
TIPS = {
    "sadness": "One small step that often helps is behavioural activation: pick one gentle activity today (a short walk, "
               "a call to a friend, or a favourite meal) and do it even if you don't feel like it.",
    "fear": "Try a grounding exercise. Breathe in for 4 seconds, hold for 4, and out for 6, repeating 5 times. "
            "Then name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell and 1 you can taste.",
    "anger": "Give yourself a pause before reacting. A short walk, slow breathing, or writing down what you feel "
             "can lower the intensity so you can respond calmly.",
    "joy": "Take a moment to note what contributed to this good feeling so you can return to it on harder days.",
    "surprise": "Give yourself time to process. Writing down your thoughts can help you sort out how you feel.",
    "disgust": "Step away from the source if you can, and do something that helps you reset, like fresh air or water.",
    "neutral": "A regular routine of sleep, meals, movement and time with people you trust supports mental well-being.",
}
FOLLOW_UPS = ["Would you like to tell me more about what happened?",
              "What feels like the hardest part of this right now?",
              "How long have you been feeling this way?",
              "What usually helps you when things get like this?"]


class Responder:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def compose(self, emotion, intent, turn_no=1):
        emotion = emotion if emotion in EMPATHY else "neutral"
        if intent == "greeting":
            return ("Hello! I'm here to listen and support you. How are you feeling today? "
                    "You can talk to me about anything, or type **check-in** for a short well-being screening.")
        if intent == "gratitude":
            return "You're very welcome. I'm glad to be here for you. Is there anything else on your mind?"
        if intent == "goodbye":
            return ("Take care of yourself. Remember you can come back any time, and that speaking with a counsellor "
                    "is always a good step if things feel heavy.")
        if intent == "advice":
            return f"{self.rng.choice(EMPATHY[emotion])}\n\n{TIPS[emotion]}\n\n{self.rng.choice(FOLLOW_UPS)}"
        text = self.rng.choice(EMPATHY[emotion])
        if emotion != "joy" and turn_no % 3 == 0:
            text += f"\n\n{TIPS[emotion]}"
        return f"{text}\n\n{self.rng.choice(FOLLOW_UPS)}"


# =============================================================================
# QUESTION ENGINE  (adaptive PHQ-9 / GAD-7)
# =============================================================================
PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading or studying",
    "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or "
    "restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]
GAD7_ITEMS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]
ANSWER_LABELS = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}
_ANSWER_WORDS = [(r"not at all|never|none", 0), (r"several days|sometimes|a few days", 1),
                 (r"more than half|often|most days", 2), (r"nearly every day|every day|always", 3)]


def parse_answer(text):
    """Convert typed text to 0-3, or None if it cannot be understood."""
    t = normalize(text)
    if re.fullmatch(r"[0-3]", t):
        return int(t)
    for pat, val in _ANSWER_WORDS:
        if re.search(pat, t):
            return val
    return None


class ScreeningSession:
    """Adaptive questioning.

    1. Core items: PHQ-2 (items 1-2) and GAD-2 (items 1-2).
    2. If PHQ-2 >= 3 the remaining PHQ-9 items 3-8 are asked; if GAD-2 >= 3 the remaining GAD-7 items 3-7 are asked.
    3. PHQ-9 item 9 (thoughts of self-harm) is ALWAYS asked, as a safety item.
    Items that are not asked are treated as not assessed (score contribution 0), so scores are lower bounds.
    """

    def __init__(self):
        self.answers = {"phq": {}, "gad": {}}
        self.queue = [("phq", 0), ("phq", 1), ("gad", 0), ("gad", 1)]
        self.stage = "core"
        self.current = None
        self.answered = 0
        self._advance()

    def _sum(self, scale, idxs):
        return sum(self.answers[scale].get(i, 0) for i in idxs)

    def _advance(self):
        if not self.queue and self.stage == "core":
            self.stage = "follow"
            if self._sum("phq", (0, 1)) >= 3:
                self.queue += [("phq", i) for i in range(2, 8)]
            if self._sum("gad", (0, 1)) >= 3:
                self.queue += [("gad", i) for i in range(2, 7)]
            self.queue.append(("phq", 8))
        self.current = self.queue.pop(0) if self.queue else None

    @property
    def finished(self):
        return self.current is None

    def question(self):
        if self.finished:
            return None
        scale, idx = self.current
        items = PHQ9_ITEMS if scale == "phq" else GAD7_ITEMS
        return {"scale": scale, "index": idx, "number": self.answered + 1, "text": items[idx],
                "is_safety_item": scale == "phq" and idx == 8}

    def answer(self, value):
        if self.finished:
            raise RuntimeError("screening already finished")
        value = int(value)
        if value not in ANSWER_LABELS:
            raise ValueError("answer must be 0-3")
        scale, idx = self.current
        self.answers[scale][idx] = value
        self.answered += 1
        self._advance()

    def results(self):
        phq = sum(self.answers["phq"].values())
        gad = sum(self.answers["gad"].values())
        return {"phq9": phq, "gad7": gad, "item9": self.answers["phq"].get(8, 0),
                "phq_items_asked": len(self.answers["phq"]), "gad_items_asked": len(self.answers["gad"])}


# =============================================================================
# SCORING MODULE (severity bands) AND RISK CLASSIFIER
# =============================================================================
def phq9_severity(score):
    for upper, label in [(4, "minimal"), (9, "mild"), (14, "moderate"), (19, "moderately severe"), (27, "severe")]:
        if score <= upper:
            return label
    return "severe"


def gad7_severity(score):
    for upper, label in [(4, "minimal"), (9, "mild"), (14, "moderate"), (21, "severe")]:
        if score <= upper:
            return label
    return "severe"


def classify_risk(phq9, gad7, item9=0, crisis_flag=False):
    """Return (risk_level, reasons). Levels: Low, Moderate, High (matches Table 4.2 of the report)."""
    reasons = []
    if crisis_flag:
        reasons.append("crisis language detected")
    if item9 > 0:
        reasons.append("reported thoughts of self-harm (PHQ-9 item 9)")
    if phq9 >= 15:
        reasons.append("PHQ-9 score of 15 or above")
    if gad7 >= 15:
        reasons.append("GAD-7 score of 15 or above")
    if reasons:
        return "High", reasons
    if phq9 >= 10:
        reasons.append("PHQ-9 score between 10 and 14")
    if gad7 >= 10:
        reasons.append("GAD-7 score between 10 and 14")
    if reasons:
        return "Moderate", reasons
    return "Low", ["PHQ-9 and GAD-7 scores below 10"]


# =============================================================================
# OUTPUT PATHWAYS
# =============================================================================
def match_resources(risk, phq9, gad7, directory=None):
    """Therapist referral: rank counselling resources by relevance to the screening outcome."""
    directory = directory or config.COUNSELLING_DIRECTORY
    wanted = {"general"}
    if phq9 >= 10:
        wanted.add("depression")
    if gad7 >= 10:
        wanted.add("anxiety")
    if risk == "High":
        wanted.add("crisis")
    scored = sorted(directory, key=lambda r: (-len(wanted & set(r["focus"])), r["name"]))
    return [r for r in scored if wanted & set(r["focus"])][:3]


def build_result_text(res, risk, reasons):
    phq, gad = res["phq9"], res["gad7"]
    partial = res["phq_items_asked"] < 9 or res["gad_items_asked"] < 7
    text = ("### Your screening summary\n"
            f"- **Depression screen (PHQ-9):** {phq} out of 27 ({phq9_severity(phq)} range)\n"
            f"- **Anxiety screen (GAD-7):** {gad} out of 21 ({gad7_severity(gad)} range)\n"
            f"- **Support level suggested:** {risk} ({'; '.join(reasons)})\n\n")
    if partial:
        text += ("_Because your first answers suggested fewer symptoms, some questions were skipped, "
                 "so these scores may be slightly lower than a full questionnaire._\n\n")
    text += ("**Important:** this is a screening only. It is **not a diagnosis**. "
             "Only a qualified professional can assess your mental health.\n\n")
    if risk == "Low":
        text += ("Your answers suggest you are coping reasonably at the moment. Keep up healthy habits: regular sleep, "
                 "meals, movement and talking to people you trust. You can check in again any time.")
    elif risk == "Moderate":
        text += ("Your answers suggest it could help to talk to a human professional. "
                 "I've listed counselling options below, and you can request an appointment.")
    else:
        text += ("Your answers suggest you may need support from a person soon. "
                 "Please use the contacts shown and consider booking an appointment below.")
    return text


# Convenience helper used by tests and the app
def screening_to_risk(session, crisis_flag=False):
    res = session.results()
    risk, reasons = classify_risk(res["phq9"], res["gad7"], res["item9"], crisis_flag)
    return res, risk, reasons
