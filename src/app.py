
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Find project folder and load the saved model
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "models" / "hospital_billing_model.pkl"

model = joblib.load(model_path)

# Page settings
st.set_page_config(
    page_title="Hospital Billing Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Billing Amount Prediction")
st.write(
    "Enter the patient and treatment details "
    "to estimate the hospital bill."
)

# Patient details
st.header("Patient Details")

age = st.number_input(
    "Age", min_value=1, max_value=120, value=35
)

gender = st.selectbox(
    "Gender", ["Male", "Female"]
)

blood_type = st.selectbox(
    "Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
)

medical_condition = st.selectbox(
    "Medical Condition",
    ["Arthritis", "Asthma", "Cancer", "Diabetes",
     "Hypertension", "Obesity"]
)

insurance = st.selectbox(
    "Insurance Provider",
    ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"]
)

admission_type = st.selectbox(
    "Admission Type", ["Elective", "Emergency", "Urgent"]
)

medication = st.selectbox(
    "Medication",
    ["Aspirin", "Ibuprofen", "Lipitor", "Paracetamol", "Penicillin"]
)

test_results = st.selectbox(
    "Test Results", ["Normal", "Abnormal", "Inconclusive"]
)

# Treatment details
st.header("Treatment Details")

treatment_type = st.selectbox(
    "Treatment Type",
    ["Medication", "Surgery", "Therapy", "Diagnostic"]
)

room_type = st.selectbox(
    "Room Type", ["General", "Private", "Semi-Private", "ICU"]
)

length_of_stay = st.number_input(
    "Length of Stay (days)",
    min_value=1, max_value=365, value=3
)

icu_days = st.number_input(
    "ICU Days", min_value=0, max_value=365, value=0
)

number_of_lab_tests = st.number_input(
    "Number of Lab Tests",
    min_value=0, max_value=500, value=5
)

# Predict the bill
if st.button("Predict Hospital Bill"):
    if icu_days > length_of_stay:
        st.error("ICU days cannot exceed the total length of stay.")
    else:
        patient_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Blood Type": blood_type,
            "Medical Condition": medical_condition,
            "Insurance Provider": insurance,
            "Admission Type": admission_type,
            "Medication": medication,
            "Test Results": test_results,
            "Treatment Type": treatment_type,
            "Room Type": room_type,
            "Length of Stay": length_of_stay,
            "ICU Days": icu_days,
            "Number of Lab Tests": number_of_lab_tests
        }])

        predicted_bill = model.predict(patient_data)[0]

        st.success(
            f"Estimated Hospital Bill: ₹{max(0, predicted_bill):,.2f}"
        )

        st.caption(
            "Educational estimate based on synthetic data; "
            "not an actual hospital quotation."
        )
