Movie Review Sentiment Analysis (BERT):=
This project uses BERT Transformer to analyze the sentiment of movie reviews — whether they are positive or negative.
It also includes a Streamlit web app for real-time prediction.

Project Overview:=
The goal of this project is to predict the sentiment of a given movie review using BERT (Bidirectional Encoder Representations from Transformers), 
a state-of-the-art NLP model from Google.

Dataset:=
Dataset Used: IMDB Movie Reviews Dataset
The dataset contains 50,000 movie reviews, equally divided between positive and negative sentiments.

Data Preprocessing:=
Converted text to lowercase
Removed HTML tags, URLs, punctuation, and numbers
Expanded short forms (e.g., can’t → cannot)
Removed stopwords and handled emojis
Tokenized and lemmatized the text
Split the data into 80% training and 20% testing

Model Used:=
Model Name: bert-base-uncased (fine-tuned for sentiment classification)
Fine-tuned using the Transformers library by Hugging Face.
Achieved an accuracy of 91.9% on the test data.

Technologies Used:=
Python
Transformers (Hugging Face)
PyTorch
Pandas
NumPy
Streamlit

Streamlit Web App:=
A user-friendly web app was created using Streamlit to allow users to test the model in real-time.
https://bertsentimentanalysis-zuq2wkvazye7sato2pjjuf.streamlit.app/

Installation:=
If you want to run this app locally:
# Clone this repository
git clone https://github.com/Pranathi2307/Bert_sentiment_analysis.git
cd Bert_sentiment_analysis
# Install dependencies
pip install -r requirements.txt
# Run the app
streamlit run app.py

