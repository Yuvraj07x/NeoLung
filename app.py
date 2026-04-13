import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_model.pkl')

# Title
st.title("Lung Cancer Risk Prediction")

# Description
st.write("This app predicts the risk of lung cancer based on various symptoms and factors.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
yellow_fingers = st.selectbox("Do you have yellow fingers?", ["No", "Yes"])
anxiety = st.selectbox("Do you have anxiety?", ["No", "Yes"])
peer_pressure = st.selectbox("Do you experience peer pressure?", ["No", "Yes"])
chronic_disease = st.selectbox("Do you have any chronic disease?", ["No", "Yes"])
fatigue = st.selectbox("Do you feel fatigued?", ["No", "Yes"])
allergy = st.selectbox("Do you have allergies?", ["No", "Yes"])
wheezing = st.selectbox("Do you experience wheezing?", ["No", "Yes"])
alcohol_consuming = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
coughing = st.selectbox("Do you have coughing?", ["No", "Yes"])
shortness_of_breath = st.selectbox("Do you have shortness of breath?", ["No", "Yes"])
swallowing_difficulty = st.selectbox("Do you have difficulty swallowing?", ["No", "Yes"])
chest_pain = st.selectbox("Do you have chest pain?", ["No", "Yes"])

# Map inputs to encoded values
gender_val = 1 if gender == "Male" else 2
smoking_val = 1 if smoking == "Yes" else 2
yellow_fingers_val = 1 if yellow_fingers == "Yes" else 2
anxiety_val = 1 if anxiety == "Yes" else 2
peer_pressure_val = 1 if peer_pressure == "Yes" else 2
chronic_disease_val = 1 if chronic_disease == "Yes" else 2
fatigue_val = 1 if fatigue == "Yes" else 2
allergy_val = 1 if allergy == "Yes" else 2
wheezing_val = 1 if wheezing == "Yes" else 2
alcohol_consuming_val = 1 if alcohol_consuming == "Yes" else 2
coughing_val = 1 if coughing == "Yes" else 2
shortness_of_breath_val = 1 if shortness_of_breath == "Yes" else 2
swallowing_difficulty_val = 1 if swallowing_difficulty == "Yes" else 2
chest_pain_val = 1 if chest_pain == "Yes" else 2

# Create input array
input_data = np.array([[gender_val, age, smoking_val, yellow_fingers_val, anxiety_val,
                       peer_pressure_val, chronic_disease_val, fatigue_val, allergy_val,
                       wheezing_val, alcohol_consuming_val, coughing_val, shortness_of_breath_val,
                       swallowing_difficulty_val, chest_pain_val]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    risk_prob = probabilities[0]  # Probability of cancer (class 1)
    
    if prediction == 1:
        result = "High Risk of Lung Cancer"
        color = "red"
    else:
        result = "Low Risk of Lung Cancer"
        color = "green"
    
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Lung Cancer: {risk_prob:.2%}")
    st.write(f"Probability of No Lung Cancer: {probabilities[1]:.2%}")
    
    # Optional: Display a progress bar or gauge
    st.progress(float(risk_prob))
    st.write("Risk Level:")
    if risk_prob > 0.7:
        st.error("High Risk")
    elif risk_prob > 0.3:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")