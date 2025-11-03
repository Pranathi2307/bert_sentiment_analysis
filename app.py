import streamlit as st
from transformers import pipeline

# Load BERT model directly from Hugging Face
@st.cache_resource
def load_model():
    # Using a lightweight pre-trained model for sentiment analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis (BERT)")
st.write("Enter your movie review below to analyze its sentiment:")

user_input = st.text_area("Your review:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        result = model(user_input)[0]
        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            st.success(f"🌟 Sentiment: {label} (Confidence: {score:.2f})")
        else:
            st.error(f"😞 Sentiment: {label} (Confidence: {score:.2f})")
