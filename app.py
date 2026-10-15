import streamlit as st
import joblib
import re

st.set_page_config(page_title="Gen Z Slang Sentiment", page_icon="💬")
st.title("💬 Gen Z Slang Sentiment Analyzer")
st.write("Enter a message and the model will predict whether its sentiment is positive or negative.")

slang = {
    "fr": "for real",
    "ngl": "not gonna lie",
    "rn": "right now",
    "idk": "i do not know",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "lowkey": "somewhat",
    "highkey": "very",
    "no cap": "truthfully",
    "bussin": "amazing",
    "fire": "amazing",
    "slay": "great",
    "ate": "did great",
    "it's giving": "it is",
    "mid": "mediocre",
    "cringe": "embarrassing",
    "sus": "suspicious",
    "l take": "bad opinion",
    "big l": "big loss",
    "big w": "big win",
    "w": "win",
    "l": "loss",
    "hits different": "feels special",
    "not it": "bad",
    "flop": "failure",
    "vibe": "good feeling"
}

def normalize_text(text):
    text = text.lower()
    for phrase in sorted(slang, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", slang[phrase], text)
    return text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

message = st.text_area("Your message", placeholder="Example: no cap, that outfit is fire 🔥")

if st.button("Analyze sentiment"):
    if message.strip():
        normalized = normalize_text(message)
        features = vectorizer.transform([normalized])
        prediction = model.predict(features)[0]
        if prediction == "positive":
            st.success("Positive sentiment 😊")
        else:
            st.error("Negative sentiment 🙁")
        st.caption("Slang-normalized text: " + normalized)
        st.caption("This is a small beginner model, so slang, sarcasm, and context may be misunderstood.")
    else:
        st.warning("Please enter a message first.")
