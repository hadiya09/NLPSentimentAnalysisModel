import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Sentiment Analysis")
st.write("Enter a sentence to check its sentiment.")

text = st.text_input("Enter text")

if st.button("Analyze"):
    if text:
        words = vectorizer.transform([text])
        result = model.predict(words)[0]
        st.write("Sentiment:", result)
    else:
        st.write("Please enter some text.")
