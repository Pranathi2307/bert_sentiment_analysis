import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the trained model and tokenizer
model_path = "bert_sentiment_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

st.title("🎭 Sentiment Analysis using BERT")

# User input
user_input = st.text_area("Enter your review:", placeholder="Type your movie review here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()

        # Map output to sentiment
        sentiment = "Positive 😀" if pred == 1 else "Negative 😔"
        st.success(f"Prediction: {sentiment}")
