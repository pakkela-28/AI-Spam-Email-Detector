# 📧 AI Spam Email Detector

An AI-based Spam Email Detector that uses Machine Learning to classify email messages as **Spam** or **Not Spam**.

## 🚀 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to analyze email messages and predict whether an email is spam or not spam.

The application provides a simple web interface where users can enter an email message and get a prediction with a confidence percentage.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Git & GitHub

## 🤖 Machine Learning

The project uses:

- **TF-IDF Vectorizer** – Converts email text into numerical features.
- **Multinomial Naive Bayes** – Classifies the email as Spam or Not Spam.

The trained model is saved as:

```text
model/spam_detector.pkl