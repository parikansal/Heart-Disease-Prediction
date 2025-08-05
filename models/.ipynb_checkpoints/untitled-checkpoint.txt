import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("💓 Heart Disease Prediction App")

st.write("Fill in the patient details to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Mapping input to numerical
sex_map = {"Male": 1, "Female": 0}
cp_map = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
fbs_map = {"Yes": 1, "No": 0}
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}

input_data = np.array([[age, 
                        sex_map[sex], 
                        cp_map[chest_pain], 
                        resting_bp, 
                        cholesterol,
                        fbs_map[fasting_bs],
                        ecg_map[rest_ecg],
                        max_hr,
                        exang_map[exercise_angina],
                        oldpeak,
                        slope_map[st_slope]]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️ The person is likely to have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")
