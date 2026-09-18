import streamlit as st
import pickle

# Load the trained model
with open("model/spam_detector.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.title("📧 AI Spam Email Detector")

st.write("Enter an email message below and the AI will predict whether it is spam.")

# Email input
email = st.text_area(
    "Enter your email message:",
    placeholder="Example: Congratulations! You won a free prize..."
)

# Check button
if st.button("Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:
        # Make prediction
        prediction = model.predict([email])[0]

        # Get confidence
        probabilities = model.predict_proba([email])[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")

        st.write(f"**Confidence:** {confidence:.2f}%")