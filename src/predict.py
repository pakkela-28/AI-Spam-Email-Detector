import pickle

# Load the trained model
with open("model/spam_detector.pkl", "rb") as file:
    model = pickle.load(file)

print("=== AI Spam Email Detector ===")

email = input("\nEnter your email message: ")

# Make prediction
prediction = model.predict([email])[0]

# Get prediction probabilities
probabilities = model.predict_proba([email])[0]

# Get confidence percentage
confidence = max(probabilities) * 100

# Display result
if prediction == "spam":
    print("\n🚨 Prediction: SPAM")
else:
    print("\n✅ Prediction: NOT SPAM")

print(f"Confidence: {confidence:.2f}%")