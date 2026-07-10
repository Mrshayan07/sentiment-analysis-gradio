from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import gradio as gr

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


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your text here..."
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence")
    ],
    title="😊 Sentiment Analysis using DistilBERT",
    description="Predict whether a sentence is Positive or Negative using a Hugging Face Transformer model."
)

demo.launch()