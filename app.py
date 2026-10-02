"""Mental Health Chatbot prototype (Streamlit front-end = the "Chat interface" of the Tier 3 architecture).

Run:  streamlit run app.py
"""
import hmac

import pandas as pd
import streamlit as st

import config
from engine import (ANSWER_LABELS, MOOD_LABELS, Database, EmotionDetector, Responder, ScreeningSession,
                    build_result_text, classify_intent, crisis_message, is_crisis, match_resources,
                    parse_answer, screening_to_risk)

st.set_page_config(page_title=config.APP_TITLE, page_icon="💬", layout="centered")


@st.cache_resource
def get_db():
    return Database(config.DB_PATH)


@st.cache_resource(show_spinner="Loading language model (the first run can take a few minutes)...")
def get_detector():
    return EmotionDetector(config.EMOTION_BACKEND, config.EMOTION_MODEL)


db = get_db()
detector = get_detector()
responder = Responder()
S = st.session_state

WELCOME = ("Hello, I'm your supportive companion. This is a private, anonymous space where you can talk about how "
           "you're feeling. I can listen, suggest coping ideas, track your mood, and offer a short well-being "
           "check-in (type **check-in**). If you ever need a human, I'll point you to one.\n\nHow are you feeling today?")


# ----------------------------------------------------------------------------- helpers
def bot_say(content, kind="normal", emotion=None):
    S.messages.append({"role": "assistant", "content": content, "kind": kind})
    db.log_message(S.session_id, "assistant", content, emotion)


def user_say(content, emotion=None):
    S.messages.append({"role": "user", "content": content, "kind": "normal"})
    db.log_message(S.session_id, "user", content, emotion)


def new_session():
    if S.get("session_id"):
        db.end_session(S["session_id"])
    S.session_id = db.create_session()
    S.messages = []
    S.mode = "chat"
    S.screening = None
    S.show_referral = False
    S.neg_count = 0
    S.offered = False
    S.turns = 0
    S.last_result = None
    S.crisis_seen = False
    S.toast = None
    bot_say(WELCOME)


# ----------------------------------------------------------------------------- output pathways
def trigger_crisis(text, source="chat"):
    """Crisis escalation: interrupt dialogue, show hotline info, log a human alert."""
    was_screening = S.mode == "screening"
    S.crisis_seen = True
    S.mode = "chat"
    S.screening = None
    db.create_alert(S.session_id, source, text)
    bot_say(crisis_message(), kind="crisis")
    if was_screening:
        bot_say("I've paused the check-in. You can restart it any time you feel ready.")
    S.show_referral = True


def finish_screening():
    """Scoring module -> risk classifier -> secure database -> output pathway."""
    res, risk, reasons = screening_to_risk(S.screening, crisis_flag=S.crisis_seen)
    db.save_screening(S.session_id, res["phq9"], res["gad7"], risk)
    S.last_result = {"res": res, "risk": risk, "reasons": reasons}
    S.mode = "chat"
    S.screening = None
    bot_say(build_result_text(res, risk, reasons))                      # Screening result
    if risk in ("Moderate", "High"):
        S.show_referral = True                                           # Therapist referral
    if risk == "High":
        trigger_crisis("(screening) " + "; ".join(reasons), source="screening")  # Crisis escalation


# ----------------------------------------------------------------------------- question engine
def ask_current():
    q = S.screening.question()
    kind = "depression" if q["scale"] == "phq" else "anxiety"
    if q["is_safety_item"]:
        intro = "This last question is important, and it's okay to answer honestly.\n\n"
    else:
        intro = ""
    bot_say(f"{intro}**Question {q['number']}** ({kind} screen)\n\nOver the **last 2 weeks**, how often have you "
            f"been bothered by: **{q['text']}**?")


def start_screening():
    if S.mode == "screening":
        return
    S.screening = ScreeningSession()
    S.mode = "screening"
    bot_say("Okay, let's do a short, private check-in. I'll ask a few questions about the last 2 weeks. "
            "It is a screening, **not** a diagnosis, and you can stop any time by typing something else.")
    ask_current()


def submit_answer(value):
    S.screening.answer(value)
    if S.screening.finished:
        finish_screening()
    else:
        ask_current()


def answer_cb(value):
    user_say(f"{value} - {ANSWER_LABELS[value]}")
    submit_answer(value)


def show_referral_cb():
    S.show_referral = True
    bot_say("Of course. I've listed counselling options below. You can request an appointment there.")


def log_mood_cb():
    v = int(S.mood_value)
    db.log_mood(S.session_id, v)
    S.toast = f"Mood logged: {MOOD_LABELS[v]}"
    if v == 1:
        bot_say("Thank you for logging that. It sounds like today is really hard. Would you like to talk about it, "
                "or take the short check-in?")


# ----------------------------------------------------------------------------- chat handling
def handle_user_text(text):
    text = (text or "").strip()
    if not text:
        return
    if is_crisis(text):                       # crisis handling always takes priority
        user_say(text, "crisis")
        trigger_crisis(text)
        return
    if S.mode == "screening":
        user_say(text)
        value = parse_answer(text)
        if value is None:
            bot_say("Please choose one of the answer buttons below, or type a number from 0 to 3 "
                    "(0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day).")
        else:
            submit_answer(value)
        return
    emotion, _conf = detector.detect(text)
    intent = classify_intent(text)
    user_say(text, emotion)
    S.turns += 1
    if emotion in ("sadness", "fear") and intent in ("venting", "advice"):
        S.neg_count += 1
    if intent == "screening_request":
        start_screening()
        return
    if intent == "referral_request":
        show_referral_cb()
        return
    if intent == "mood_request":
        bot_say("You can log how you feel with the **mood slider** in the sidebar and see your trend under "
                "**Mood dashboard**.")
        return
    reply = responder.compose(emotion, intent, S.turns)
    if S.neg_count >= 3 and not S.offered and intent == "venting":
        reply += ("\n\nYou've shared a lot of difficult feelings. If you'd like, I can guide you through a short "
                  "well-being check-in. Just type **check-in**.")
        S.offered = True
    bot_say(reply, emotion=emotion)


# ----------------------------------------------------------------------------- UI pieces
def render_messages():
    for m in S.messages:
        with st.chat_message(m["role"]):
            if m["kind"] == "crisis":
                st.error(m["content"])
            else:
                st.markdown(m["content"])


def render_answer_buttons():
    n = S.screening.answered
    st.write("**Choose your answer:**")
    for v in range(4):
        st.button(f"{v} - {ANSWER_LABELS[v]}", key=f"ans_{n}_{v}", on_click=answer_cb, args=(v,))


def render_referral_panel():
    res = S.last_result
    risk = res["risk"] if res else "Low"
    phq = res["res"]["phq9"] if res else 0
    gad = res["res"]["gad7"] if res else 0
    matches = match_resources(risk, phq, gad)
    with st.expander("Counselling options and appointment request", expanded=True):
        if not matches:
            st.info("No counselling services are configured yet. Please edit COUNSELLING_DIRECTORY in config.py.")
            return
        for r in matches:
            st.markdown(f"**{r['name']}** ({r['type']})  \nAvailability: {r['availability']}  \nContact: {r['contact']}")
        options = {r["id"]: r["name"] for r in matches}
        with st.form("referral_form", clear_on_submit=True):
            rid = st.selectbox("Choose a service", list(options), format_func=lambda k: options[k])
            when = st.text_input("Preferred day and time (optional)")
            contact = st.text_input("Phone or email (optional; leave blank to stay anonymous)")
            submitted = st.form_submit_button("Request appointment")
        if submitted:
            db.create_referral(S.session_id, rid, when.strip(), contact.strip())
            bot_say(f"Your appointment request for **{options[rid]}** has been recorded. A counsellor can follow up "
                    "if you left contact details. Please also use the contacts above if you need help sooner.")
            st.success("Request recorded. Thank you for reaching out.")


def render_mood_dashboard():
    st.header("Mood dashboard")
    entries = db.get_mood_entries(S.session_id)
    if entries:
        df = pd.DataFrame(entries)
        df["entry"] = range(1, len(df) + 1)
        st.subheader("Mood over this session")
        st.line_chart(df.set_index("entry")["mood_value"])
        st.dataframe(df[["timestamp", "mood_label"]])
    else:
        st.info("No mood entries yet. Use the mood slider in the sidebar and press **Log mood**.")
    counts = db.get_emotion_counts(S.session_id)
    if counts:
        st.subheader("Emotions detected in your messages")
        st.bar_chart(pd.Series(counts))
    shots = db.get_screenings(S.session_id)
    if shots:
        st.subheader("Check-in results")
        st.dataframe(pd.DataFrame(shots)[["timestamp", "phq9_score", "gad7_score", "risk_level"]])


def render_counsellor_view():
    st.header("Counsellor view")
    expected = config.COUNSELLOR_PASSWORD or "counsellor"
    if not config.COUNSELLOR_PASSWORD:
        st.warning("Development password in use. Set the COUNSELLOR_PASSWORD environment variable before real use.")
    pw = st.text_input("Password", type="password")
    if not pw:
        return
    if not hmac.compare_digest(pw.encode(), expected.encode()):
        st.error("Incorrect password.")
        return
    alerts = db.list_alerts()
    referrals = db.list_referrals()
    screens = db.get_screenings()
    c1, c2, c3 = st.columns(3)
    c1.metric("Open crisis alerts", sum(1 for a in alerts if a["status"] == "open"))
    c2.metric("Referral requests", len(referrals))
    c3.metric("Screenings", len(screens))
    st.subheader("Crisis alerts (human follow-up)")
    if not alerts:
        st.info("No alerts.")
    for a in alerts:
        st.markdown(f"**Alert #{a['alert_id']}** | {a['timestamp']} | session `{a['session_id']}` | "
                    f"source: {a['source']} | status: **{a['status']}**  \n> {a['snippet']}")
        if a["status"] == "open":
            st.button("Mark reviewed", key=f"rev_{a['alert_id']}", on_click=db.set_alert_status,
                      args=(a["alert_id"], "reviewed"))
    st.subheader("Appointment requests")
    if referrals:
        st.dataframe(pd.DataFrame(referrals))
    else:
        st.info("No requests yet.")
    if screens:
        st.subheader("Screening outcomes (anonymous)")
        st.bar_chart(pd.Series([s["risk_level"] for s in screens]).value_counts())


# ----------------------------------------------------------------------------- main
if "session_id" not in S:
    new_session()

with st.sidebar:
    st.title("💬 Support Chatbot")
    view = st.radio("View", ["Chat", "Mood dashboard", "Counsellor view"])
    st.divider()
    st.subheader("How are you feeling?")
    st.select_slider("Mood", options=[1, 2, 3, 4, 5], value=3, key="mood_value",
                     format_func=lambda v: MOOD_LABELS[v])
    st.button("Log mood", on_click=log_mood_cb)
    st.divider()
    st.button("Start check-in (PHQ-9 / GAD-7)", on_click=start_screening, disabled=(S.mode == "screening"))
    st.button("Find a counsellor", on_click=show_referral_cb)
    st.button("New session", on_click=new_session)
    st.divider()
    st.info(config.DISCLAIMER)
    st.caption(f"NLP engine: {detector.backend}" + ("" if detector.backend == "transformer" else " (built-in fallback)"))

if S.get("toast"):
    st.toast(S.toast)
    S.toast = None

if view == "Chat":
    st.title(config.APP_TITLE)
    st.caption(config.APP_SUBTITLE)
    prompt = st.chat_input("Type how you're feeling...")
    if prompt:
        handle_user_text(prompt)
    render_messages()
    if S.mode == "screening" and S.screening is not None and not S.screening.finished:
        render_answer_buttons()
    if S.show_referral:
        render_referral_panel()
elif view == "Mood dashboard":
    render_mood_dashboard()
else:
    render_counsellor_view()
