import streamlit as st
import numpy as np
import joblib

# Load the machine learning model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please upload 'model.pkl'.")
        st.stop()

model = load_model()

# App Title
st.title("Toxicity Prediction Dashboard")
st.markdown("""
Analyze and predict toxicity levels in manholes using machine learning. 
Enter the required features below to get a prediction.
""")

# Input Section
st.subheader("Enter Features")
num_features = 4  # Set the number of features required by the model
features = []

# Dynamically generate input fields for 4 features
for i in range(num_features):
    value = st.number_input(f"Feature {i + 1}", value=0.0, step=0.01)
    features.append(value)

# Prediction Button
if st.button("Predict"):
    try:
        # Prepare input for the model
        input_features = np.array(features).reshape(1, -1)  # Convert list to numpy array
        prediction = model.predict(input_features)
        probabilities = model.predict_proba(input_features)
        confidence = np.max(probabilities) * 100  # Get the highest probability
        labels = ["Safe", "Moderate", "Dangerous"]  # Update these with actual labels
        predicted_label = labels[prediction[0]]

        # Display Results
        st.success(f"### Prediction: {predicted_label}")
        st.write(f"### Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")
