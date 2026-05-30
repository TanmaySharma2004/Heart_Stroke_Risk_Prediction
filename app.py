import streamlit as st
import pandas as pd
import joblib

# Load model + scaler + columns
model = joblib.load('knn_heart_model.pkl')  
scaler = joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

st.title("Heart Stroke Risk Prediction by Tanmay_Sharma 💖")
st.markdown("Enter details below:")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", 1, 120, 30)

sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120", ["True", "False"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

max_hr = st.number_input("Max Heart Rate", 60, 220, 150)

angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- ENCODING ---------------- #

sex = 1 if sex == "Male" else 0

chest_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
chest_pain_type = chest_map[chest_pain_type]

fasting_bs = 1 if fasting_bs == "True" else 0

ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
resting_ecg = ecg_map[resting_ecg]

angina = 1 if angina == "Yes" else 0

slope_map = {"Up": 0, "Flat": 1, "Down": 2}
st_slope = slope_map[st_slope]

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_hr,
        "exercise_induced_angina": angina,
        "st_depression": st_depression,
        "st_slope": st_slope
    }])

    # Align columns properly
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Stroke! Please consult a doctor immediately.")
    else:
        st.success("✅ Low Risk of Heart Stroke.")