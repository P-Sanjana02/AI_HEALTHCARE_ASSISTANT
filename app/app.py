import streamlit as st
import pandas as pd
import joblib
import os
import sys

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_PATH)

# IMPORTS

from symptom_extractor import extract_symptoms
from predict import predict_disease
from severity import get_severity
from precautions import get_precautions
from explainability import explain_disease, get_shap_explanation
from report_generator import generate_report

# PAGE CONFIG

st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")

st.title("🩺 AI Healthcare Assistant")
st.info("AI-based disease prediction system (NOT a medical diagnosis)")
st.markdown("---")

# LOAD DATA

train_path = os.path.join(BASE_DIR, "data", "raw", "Training.csv")

df = pd.read_csv(train_path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

symptoms_list = list(df.drop("prognosis", axis=1).columns)

# LOAD LABEL ENCODER

encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
label_encoder = joblib.load(encoder_path)

# SESSION STATE

if "result" not in st.session_state:
    st.session_state.result = None

# PATIENT DETAILS

st.markdown("## 👤 Patient Details")
patient_name = st.text_input("Enter Patient Name")
age_input = st.text_input("Enter Age")

try:
    age = int(age_input) if age_input else 0
except:
    age = -1

# MEDICAL CONDITIONS

st.markdown("## 🏥 Existing Medical Conditions")
col1, col2 = st.columns(2)
medical_conditions = []

with col1:
    if st.checkbox("BP (Blood Pressure)"):
        medical_conditions.append("BP")

    if st.checkbox("Sugar / Diabetes"):
        medical_conditions.append("Diabetes")

    if st.checkbox("Gastric Problems"):
        medical_conditions.append("Gastric")

    if st.checkbox("Asthma"):
        medical_conditions.append("Asthma")

with col2:
    if st.checkbox("Thyroid"):
        medical_conditions.append("Thyroid")

    if st.checkbox("Heart Disease"):
        medical_conditions.append("Heart Disease")

    if st.checkbox("Kidney Disease"):
        medical_conditions.append("Kidney Disease")

    if st.checkbox("Liver Disease"):
        medical_conditions.append("Liver Disease")

# OTHER MEDICAL CONDITIONS

st.markdown("## 🏥 Other Medical Conditions")

other_conditions = st.text_area("Enter any other medical conditions",
    placeholder="Example: Migraine, Arthritis, PCOS")

# SYMPTOMS INPUT

st.markdown("## 🤒 Current Symptoms")
user_input = st.text_area("Enter Symptoms", "fever, cough, headache")

if st.button("Predict Disease"):

#  VALIDATION 
    if not patient_name.strip():
        st.warning("Enter patient name")
        st.stop()

    if age <= 0 or age > 120:
        st.warning("Enter valid age")
        st.stop()

    if not user_input.strip():
        st.warning("Enter symptoms")
        st.stop()

#  SYMPTOM EXTRACTION 
    user_symptoms = extract_symptoms(user_input, symptoms_list)

# PREDICTION 
    probs, top, valid, invalid, input_df = predict_disease(user_symptoms, medical_conditions, other_conditions)

    if probs is None:
        st.error("No valid symptoms detected")
        st.stop()

# RESULT
    best_index = top[0]
    best = label_encoder.inverse_transform([best_index])[0]
    confidence = float(probs[best_index]) * 100

# SHAP EXPLANATION

    shap_result = get_shap_explanation( input_df,best_index)    

    severity_level, severity_score = get_severity(valid , medical_conditions, best)
    description = explain_disease(best)
    precautions = get_precautions(best)

#  TOP PREDICTIONS  
    top_predictions = [
        (
            label_encoder.inverse_transform([i])[0],
            float(probs[i]) * 100
        )
        for i in top
    ]

#  SAVE RESULT 
    st.session_state.result = {
        "patient_name": patient_name,
        "age": age,
        "best": best,
        "confidence": confidence,
        "valid": valid,
        "invalid": invalid,
        "severity_level": severity_level,
        "severity_score": severity_score,
        "description": description,
        "precautions": precautions,
        "top": top,
        "probs": probs,
        "medical_conditions": medical_conditions,
        "other_conditions": other_conditions,
        "top_predictions": top_predictions,
        "shap_result": shap_result
    }

# OUTPUT SECTION

if st.session_state.result:
    r = st.session_state.result
    st.success(f"Patient: {r['patient_name']} | Age: {r['age']}")

# DISEASE 
    st.markdown("## 🩺 Predicted Disease")
    st.success(f"{r['best']} ({r['confidence']:.2f}%)")
    st.info(f"Model Confidence: {r['confidence']:.2f}%\n\n"
    "This is an AI model score and should not be interpreted "
    "as a confirmed medical diagnosis.")

# SEVERITY 
    st.markdown("## ⚠ Severity")
    st.warning(f"{r['severity_level']} ({r['severity_score']}/10)")

#  CONDITIONS 
    if r["medical_conditions"]:
        st.markdown("## 🏥 Medical Conditions")
        st.info(", ".join(r["medical_conditions"]))

    if r["other_conditions"]:
        st.markdown("## 🏥 Other Medical Conditions")
        st.info(r["other_conditions"])

#  SYMPTOMS 
    st.markdown("## 🧾 Symptoms")
    st.success(", ".join(r["valid"]))

    if r["invalid"]:
        st.warning(", ".join(r["invalid"]))

#  DESCRIPTION 
    st.markdown("## 🧠 Description")
    st.info(r["description"])

# SHAP EXPLANATION

    st.markdown("🔍 Why was this disease predicted?")
    st.dataframe(r["shap_result"],use_container_width=True)    

#  PRECAUTIONS 
    st.markdown("## 🛡 Precautions")

    if r["precautions"]:
        for p in r["precautions"]:
            st.write("•", p)
    else:
        st.info("No precautions available")

#  TOP PREDICTIONS 
    st.markdown("## 📊 Top Disease Predictions")

    top_df = pd.DataFrame({
        "Disease": [i[0] for i in r["top_predictions"]],
        "Probability (%)": [round(i[1], 2) for i in r["top_predictions"]]
    })

    st.dataframe(top_df, use_container_width=True)
    st.bar_chart(top_df.set_index("Disease"))

# GENERATE REPORT 
    file_path = generate_report(
        r["patient_name"],
        r["age"],
        r["best"],
        r["valid"],
        r["severity_level"],
        r["severity_score"],
        r["precautions"],
        r["description"],
        r["shap_result"],
        r["medical_conditions"],
        r["other_conditions"],
        r["top_predictions"]
    )

    with open(file_path, "rb") as f:
        st.download_button(
            "📄 Download Medical Report",
            f,
            file_name="medical_report.pdf",
            mime="application/pdf"
        )