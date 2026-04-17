# =============================================================
# Mental Health AI Chatbot (Web App)
# Hybrid architecture:
#   - Safety layer: regex-based crisis detection
#   - Intent layer: TF-IDF + Logistic Regression classifier
#   - Sentiment layer: VADER (rule-based lexicon)
#   - Generation layer: LLM (Groq / OpenAI / Together AI) — optional
# Run with: streamlit run app.py
# =============================================================

import streamlit as st
import random
import nltk
import re
import csv
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ---------- PATHS ----------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data.csv"
LOGS_PATH = BASE_DIR / "logs.csv"


# ---------- NLTK SETUP ----------
@st.cache_resource
def setup_nltk():
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("vader_lexicon", "sentiment/vader_lexicon"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    return set(stopwords.words("english")), SentimentIntensityAnalyzer()


# Words that flip meaning — must NOT be removed during cleaning
NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "nowhere", "neither", "none",
    "cannot", "cant", "wont", "isnt", "arent", "dont", "doesnt", "didnt",
    "wasnt", "werent", "hasnt", "havent", "hadnt", "wouldnt", "shouldnt", "couldnt",
}

_RAW_STOP_WORDS, VADER = setup_nltk()
STOP_WORDS = _RAW_STOP_WORDS - NEGATION_WORDS


# ---------- CRISIS DETECTION (bypasses everything for safety) ----------
CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bkilling myself\b", r"\bkill me\b",
    r"\bsuicide\b", r"\bsuicidal\b", r"\bcommit suicide\b",
    r"\bend my life\b", r"\bend it all\b", r"\btake my life\b",
    r"\bwant to die\b", r"\bwish i was dead\b", r"\bbetter off dead\b",
    r"\bharm myself\b", r"\bhurt myself\b", r"\bself.harm\b",
    r"\bcut myself\b", r"\bcutting myself\b",
    r"\bno reason to live\b", r"\bnothing to live for\b",
    r"\bdon'?t want to live\b", r"\bcan'?t go on\b",
]

CRISIS_HELPLINES = """
🆘 **Please reach out for immediate support — you don't have to go through this alone:**

- **iCall** (free counselling): **9152987821** — Mon–Sat, 8am–10pm
- **Vandrevala Foundation**: **1860-2662-345** — 24/7
- **AASRA**: **+91-9820466726** — 24/7
- **KIRAN (Govt. of India)**: **1800-599-0019** — 24/7, multilingual
- **Emergency**: **112**
"""


def is_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in CRISIS_PATTERNS)


# ---------- FALLBACK CANNED RESPONSES (when LLM disabled) ----------
RULE_RESPONSES = {
    "depression": [
        "I hear that you're going through a really heavy time. Your feelings are valid. Have you been able to share any of this with someone close to you?",
        "I'm sorry you're feeling this way. Depression can make everything feel weighted. Talking to a counsellor or someone you trust can really help — you don't have to carry it alone.",
        "Thank you for sharing something so personal. What you're feeling matters. If these feelings have lasted a while, please consider reaching out to a mental health professional.",
    ],
    "anxiety": [
        "It sounds like your mind is carrying a lot right now. Try a quick grounding exercise — name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "Anxiety can make everything feel urgent. Slow breathing — in for 4 seconds, hold for 4, out for 6 — can help settle your nervous system.",
        "Constant worry is exhausting. Would it help to talk through what's specifically on your mind right now?",
    ],
    "stress": [
        "Feeling overwhelmed is a sign you're carrying a lot. Try breaking things into smaller pieces and tackling one at a time.",
        "Burnout is real, and rest isn't a reward — it's a necessity. Even a short break, water, or a walk can reset things.",
        "Pressure from work or studies is hard. What's the one thing weighing on you most right now?",
    ],
    "loneliness": [
        "Loneliness is one of the hardest feelings to sit with. I'm glad you reached out.",
        "Feeling unseen is painful. Sometimes a small message to one person — even a short hello — can help bridge the distance.",
        "You're not as alone as it feels right now. Would you like to share what's been making you feel this way?",
    ],
    "positive": [
        "That's wonderful to hear! Hold onto that feeling.",
        "Love that for you. What's been going well?",
        "Glad you're doing okay! Keep nurturing the things that are working.",
    ],
}


# ---------- DATA + PREPROCESSING ----------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"data.csv not found at {DATA_PATH}. Place it next to app.py.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


# ---------- VADER ----------
def get_intensity(text: str) -> dict:
    return VADER.polarity_scores(text)


def intensity_label(compound: float) -> str:
    if compound <= -0.6:
        return "Very negative"
    elif compound <= -0.2:
        return "Negative"
    elif compound < 0.2:
        return "Neutral"
    elif compound < 0.6:
        return "Positive"
    else:
        return "Very positive"


# ---------- MODEL TRAINING ----------
@st.cache_resource
def train_model():
    df = load_data()
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    labels_sorted = sorted(df["label"].unique())

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred, labels=labels_sorted),
        "labels": labels_sorted,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(df),
    }
    return vectorizer, model, metrics


vectorizer, model, metrics = train_model()


def predict(text: str):
    cleaned = clean_text(text)
    if not cleaned:
        return "positive", 0.0
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    return model.classes_[idx], float(probs[idx])


# ---------- LOGGING ----------
def log_prediction(text, category, confidence, vader_compound, was_crisis, backend):
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text_length": len(text),
        "word_count": len(text.split()),
        "category": category,
        "confidence": round(confidence, 3),
        "vader_compound": round(vader_compound, 3),
        "crisis_detected": was_crisis,
        "backend": backend,
    }
    file_exists = LOGS_PATH.exists()
    with open(LOGS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


# =============================================================
# LLM BACKEND
# =============================================================

PROVIDER_CONFIGS = {
    "Groq (free)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        "signup": "https://console.groq.com",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "signup": "https://platform.openai.com",
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        ],
        "signup": "https://api.together.xyz",
    },
}

INTENT_GUIDANCE = {
    "depression": "Be warm and validating. Acknowledge their pain without rushing to fix it. Gently mention that talking to someone (friend, family, counsellor) can help if it feels right.",
    "anxiety": "Acknowledge the racing thoughts or worry. Offer ONE concrete grounding or breathing technique. Stay calm and steadying — don't pile on suggestions.",
    "stress": "Validate that they're carrying a lot. Help them break things down. Suggest small, immediate self-care steps.",
    "loneliness": "Be present and make them feel heard. Gently encourage one small connection step if appropriate.",
    "positive": "Celebrate with them. Be warm and curious. Encourage them to savor the moment.",
}

SYSTEM_PROMPT_TEMPLATE = """You are a warm, empathetic mental health support companion built for an educational student project.

CRITICAL RULES (never violate):
- You are NOT a therapist, doctor, or medical professional. Never diagnose, prescribe, or give medical advice.
- For serious or persistent concerns, gently remind the user to seek professional support.
- Keep responses concise — usually 2 to 4 sentences. Be conversational, not clinical.
- Validate feelings BEFORE offering suggestions.
- Never lecture, never moralize, never use bullet lists in replies.
- Use "I" sparingly — focus on the user.

CONTEXT FROM CLASSIFIER (for your awareness, do not mention these labels to the user):
- Detected emotional category: {category}
- Sentiment intensity (VADER compound, -1 to +1): {compound:+.2f} ({intensity_label})
- Approach guidance: {intent_guidance}

Respond naturally to the user's most recent message, taking the prior conversation into account."""


def build_system_prompt(category, compound):
    return SYSTEM_PROMPT_TEMPLATE.format(
        category=category,
        compound=compound,
        intensity_label=intensity_label(compound),
        intent_guidance=INTENT_GUIDANCE.get(category, "Listen and be supportive."),
    )


def stream_llm_response(history, category, compound, api_key, base_url, model):
    """Yields response chunks from any OpenAI-compatible chat completions endpoint."""
    system_prompt = build_system_prompt(category, compound)
    full_messages = [{"role": "system", "content": system_prompt}]

    # Last 6 messages (≈3 turns) keeps context relevant without bloating token use
    for msg in history[-6:]:
        if msg["role"] in ("user", "assistant"):
            full_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        with requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": full_messages,
                "temperature": 0.7,
                "max_tokens": 250,
                "stream": True,
            },
            stream=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                yield f"⚠️ API error ({response.status_code}). Check your API key and model name."
                return
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Connection error: {e}. Falling back to rule-based reply."


# =============================================================
# UI
# =============================================================
st.set_page_config(page_title="Mental Health Companion", page_icon="🧠", layout="centered")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🧠 Companion")
    view_mode = st.radio("View", ["💬 Chat", "📊 Analytics"], label_visibility="collapsed")

    st.divider()
    st.subheader("⚙️ Response engine")
    backend_choice = st.radio(
        "Backend",
        ["Rule-based (offline)", "LLM (online)"],
        label_visibility="collapsed",
    )

    llm_config = None
    if backend_choice == "LLM (online)":
        provider = st.selectbox("Provider", list(PROVIDER_CONFIGS.keys()))
        cfg = PROVIDER_CONFIGS[provider]
        st.caption(f"Get a free API key at {cfg['signup']}")
        api_key = st.text_input(
            "API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Stored only in your browser session",
        )
        model_name = st.selectbox("Model", cfg["models"])
        if api_key:
            st.session_state.api_key = api_key
            llm_config = {
                "api_key": api_key,
                "base_url": cfg["base_url"],
                "model": model_name,
                "provider": provider,
            }
            st.success("✓ LLM enabled")
        else:
            st.warning("Enter an API key to enable LLM mode")

    st.divider()
    st.header("📞 Resources")
    st.markdown(CRISIS_HELPLINES)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hi, I'm here to listen. How are you feeling today?"}
        ]
        st.rerun()


# ===========================
# CHAT VIEW
# ===========================
if view_mode == "💬 Chat":
    st.title("🧠 Mental Health Companion")
    backend_label = "LLM-powered" if llm_config else "Rule-based"
    st.caption(f"A safe space to share how you're feeling · *{backend_label} mode*")

    with st.expander("⚠️ Important — please read", expanded=False):
        st.markdown(
            "This chatbot is an **educational project** and **not a substitute for professional help**. "
            "If you are in crisis or struggling, please reach out to a qualified mental health professional "
            "or one of the helplines in the sidebar."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hi, I'm here to listen. How are you feeling today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type how you're feeling..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            crisis = is_crisis(user_input)
            sentiment = get_intensity(user_input)
            compound = sentiment["compound"]
            backend_used = "rule"

            if crisis:
                # Crisis ALWAYS uses canned response — never trust LLM here
                reply = (
                    "I'm really concerned about what you just shared. Your life matters, "
                    "and you deserve support from someone trained to help right now.\n\n"
                    + CRISIS_HELPLINES
                )
                category, confidence = "crisis", 1.0
                st.markdown(reply)
            else:
                category, confidence = predict(user_input)

                # VADER override layer
                if category == "positive" and compound <= -0.2:
                    category = "depression"
                elif category in ("depression", "anxiety", "stress", "loneliness") and compound >= 0.5:
                    category = "positive"

                if llm_config:
                    backend_used = llm_config["provider"]
                    reply = st.write_stream(stream_llm_response(
                        st.session_state.messages,
                        category, compound,
                        llm_config["api_key"],
                        llm_config["base_url"],
                        llm_config["model"],
                    ))
                    # If the stream returned only an error message, fall back
                    if reply.startswith("⚠️"):
                        st.info("Falling back to rule-based reply.")
                        reply = random.choice(RULE_RESPONSES[category])
                        st.markdown(reply)
                        backend_used = "rule (fallback)"
                else:
                    reply = random.choice(RULE_RESPONSES[category])
                    if confidence < 0.35:
                        reply += "\n\n*I want to make sure I understand you — could you tell me a bit more about what's going on?*"
                    if compound <= -0.7 and category not in ("depression", "positive"):
                        reply += "\n\n*Your message conveys strong emotional weight. Please be kind to yourself, and remember help is always available.*"
                    st.markdown(reply)

            with st.expander("🔬 Model insight (for the demo)"):
                st.write(f"**Backend:** `{backend_used}`")
                st.write(f"**Predicted category:** `{category}`")
                st.write(f"**Classifier confidence:** `{confidence:.2%}`")
                st.write(f"**VADER compound score:** `{compound:+.3f}` ({intensity_label(compound)})")
                st.write(f"**Crisis keyword match:** `{crisis}`")

            st.session_state.messages.append({"role": "assistant", "content": reply})
            log_prediction(user_input, category, confidence, compound, crisis, backend_used)


# ===========================
# ANALYTICS VIEW
# ===========================
else:
    st.title("📊 Model Analytics")
    st.caption("Performance evaluation and live usage statistics")

    st.subheader("Test set performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Train samples", metrics["n_train"])
    c3.metric("Test samples", metrics["n_test"])

    st.subheader("Confusion matrix (test set)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics["cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=metrics["labels"], yticklabels=metrics["labels"],
        cbar=False, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Per-class metrics")
    report_df = pd.DataFrame(metrics["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Live prediction log (anonymized)")
    st.caption("Only metadata is stored — no raw user text is logged.")
    if LOGS_PATH.exists():
        logs_df = pd.read_csv(LOGS_PATH)
        st.write(f"**Total predictions logged:** {len(logs_df)}")
        st.dataframe(logs_df.tail(25), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Category distribution**")
            st.bar_chart(logs_df["category"].value_counts())
        with col_b:
            st.write("**Crisis detections**")
            st.bar_chart(logs_df["crisis_detected"].value_counts())
        with col_c:
            if "backend" in logs_df.columns:
                st.write("**Backend used**")
                st.bar_chart(logs_df["backend"].value_counts())
    else:
        st.info("No predictions logged yet. Use the chat to start collecting data.")
