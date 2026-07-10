from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    label = model.config.id2label[prediction]
    confidence = probabilities[0][prediction].item()

    return label, f"{confidence*100:.2f}%"


import streamlit as st

st.title("😊 Sentiment Analysis using DistilBERT")

text = st.text_area("Enter your text")

if "label" not in st.session_state:
    st.session_state.label = None
    st.session_state.confidence = None

if st.button("Predict"):
    label, confidence = predict_sentiment(text)

    st.session_state.label = label
    st.session_state.confidence = confidence

    st.success(f"Sentiment: {label}")
    st.info(f"Confidence: {confidence}")

if st.button("🚩 Save Feedback"):

    if st.session_state.label is None:
        st.warning("Please predict first!")
    else:
        with open("feedback.csv", "a") as f:
            f.write(
                f"{text},{st.session_state.label},{st.session_state.confidence}\n"
            )

        st.success("Feedback saved!")