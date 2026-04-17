# 🧠 Mental Health Companion

A multi-layered conversational AI chatbot that provides preliminary mental wellness support, combining classical machine learning, lexicon-based sentiment analysis, regex-based crisis detection, and modern Large Language Models in a single hybrid pipeline.

> ⚠️ **This is an educational project, not a medical tool.** It is not a substitute for professional mental health care. If you or someone you know is in crisis, please contact a qualified mental health professional or one of the helplines listed in the app.

---

## ✨ Features

- **Four-layer hybrid pipeline** — safety, intent, sentiment, and generation are cleanly separated so each layer can be reasoned about and tested independently.
- **Regex-based crisis detection** — 21 patterns covering self-harm and suicidal ideation. Bypasses both the classifier and the LLM, returning India-specific helpline numbers immediately.
- **TF-IDF + Logistic Regression classifier** — interpretable, fast, multinomial intent classification across 5 categories (depression, anxiety, stress, loneliness, positive).
- **VADER sentiment override** — catches negation patterns ("not feeling good") that bag-of-words models structurally miss.
- **Two interchangeable response backends:**
  - **Rule-based (offline)** — curated empathic templates, no internet required
  - **LLM (online)** — streaming responses from Groq, OpenAI, or Together AI with conversation memory
- **Real-time streaming** — LLM responses appear word-by-word like ChatGPT, using Server-Sent Events.
- **Built-in analytics dashboard** — confusion matrix, per-class precision/recall/F1, and live anonymized usage logs.
- **Privacy-first logging** — only metadata is stored (length, category, sentiment score). No raw user text is ever persisted.

---

## 🏗️ Architecture

```
User input
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 1: Crisis Detection (regex)   │──► If matched: return helpline (no LLM)
└─────────────────────────────────────┘
    │ (if no match)
    ▼
┌─────────────────────────────────────┐
│ Layer 2: Intent Classifier          │
│ TF-IDF + Logistic Regression        │──► category, confidence
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 3: VADER Sentiment            │──► compound score ∈ [-1, +1]
│ (overrides classifier on negation)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 4: Response Generation        │
│  • Rule-based templates (offline)   │
│  • LLM with system prompt (online)  │
└─────────────────────────────────────┘
    │
    ▼
Streamed response + anonymized log entry
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, if `pip` isn't on your PATH:

```bash
py -m pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Or on Windows:

```bash
py -m streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 💬 Using the LLM Mode (Optional)

The app works fully offline by default with rule-based responses. To enable richer conversational responses, switch to LLM mode:

1. Get a **free Groq API key** at [console.groq.com](https://console.groq.com) (no credit card needed).
2. In the app sidebar, switch **Response Engine** to **"LLM (online)"**.
3. Pick **Groq (free)** as the provider.
4. Paste your API key.
5. Choose a model:
   - `llama-3.1-8b-instant` — fast, ~1 second responses
   - `llama-3.3-70b-versatile` — higher quality, ~2–3 second responses

Other supported providers: **OpenAI** and **Together AI** (both use the same OpenAI-compatible API format — just paste their respective API keys and pick a model).

---

## 📁 Project Structure

```
mental-health-chatbot/
├── app.py              # Streamlit application — UI, all 4 pipeline layers, analytics
├── data.csv            # 96 labeled training statements
├── requirements.txt    # Python dependencies
├── logs.csv            # Auto-generated anonymized prediction log (created on first run)
└── README.md           # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Vectorizer | scikit-learn TfidfVectorizer (unigrams + bigrams) |
| Classifier | scikit-learn LogisticRegression |
| Sentiment | NLTK VADER |
| Crisis detection | Python `re` (regex) |
| LLM client | `requests` (OpenAI-compatible chat completions API) |
| Visualization | matplotlib + seaborn |

---

## 🆘 Crisis Helplines (India)

The app surfaces these helplines whenever a crisis pattern is detected. They are also always available in the sidebar:

- **iCall** (free counselling): **9152987821** — Mon–Sat, 8am–10pm
- **Vandrevala Foundation**: **1860-2662-345** — 24/7
- **AASRA**: **+91-9820466726** — 24/7
- **KIRAN (Govt. of India)**: **1800-599-0019** — 24/7, multilingual
- **Emergency**: **112**

---

## 📊 Model Performance

Evaluated on a stratified 25% held-out test split of the 96-example dataset:

- **Test accuracy:** 87.5%
- **Macro F1:** 0.872
- **Depression class recall:** 100% (the most safety-critical category)
- **Negation handling:** 100% on hand-crafted negation probes (vs 8.3% without VADER override)
- **Crisis detection recall:** 100% by construction

Switch to the **📊 Analytics** view in the app sidebar to see the full confusion matrix, per-class metrics, and live usage statistics.

---

## 🔮 Future Work

- Expand the dataset to 2,000+ labeled examples
- Add Hindi and Hinglish support with a language detector at the front of the pipeline
- Replace TF-IDF with a sentence-transformer encoder for better semantic understanding
- Add Retrieval-Augmented Generation grounded in evidence-based mental health resources
- Deploy as a Progressive Web App with offline mode

---

## ⚠️ Important Disclaimer

This chatbot is built as an **educational project** to demonstrate hybrid ML+LLM architecture in a sensitive domain. It is **not a substitute for professional mental health care**. The responses are generated by automated systems and may be inappropriate or incorrect.

If you or someone you know is in distress, please reach out to:
- A qualified mental health professional
- One of the helplines listed above
- Emergency services (112 in India)

---

## 👤 Author

**Amlan Mishra** — Roll 23053072  
School of Computer Engineering  
Kalinga Institute of Industrial Technology (KIIT)  
Bhubaneswar, Odisha

---

## 📄 License

This project is released under the MIT License — see the LICENSE file for details. The KIIT University name and logo remain the property of KIIT University and are used here only for academic identification.
