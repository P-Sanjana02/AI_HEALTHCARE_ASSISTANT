import os
import joblib
import numpy as np
import pandas as pd
from symptom_extractor import extract_symptoms  

# BASE DIRECTORY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PATHS

RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

print("\nLoading models...")

rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

symptoms_list = rf_model.feature_names_in_.tolist()

# PREDICTION FUNCTION

def predict_disease(
        user_symptoms,
        medical_conditions=None,
        other_conditions=""
):

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=symptoms_list
    )
    valid = []
    invalid = []

# Symptom Processing

    for s in user_symptoms:

        if s in symptoms_list:
            input_df.loc[0, s] = 1
            valid.append(s)

        else:
            invalid.append(s)

    if len(valid) == 0:
        return None, None, valid, invalid , None

# Model Predictions

    rf_probs = rf_model.predict_proba(input_df)[0]
    xgb_probs = xgb_model.predict_proba(input_df)[0]
    avg_probs = (
        0.6 * rf_probs
        + 0.4 * xgb_probs
    )

    disease_names = label_encoder.inverse_transform(np.arange(len(avg_probs)))
 
    for idx, disease in enumerate(disease_names):

        disease = disease.lower()

        # Diabetes

        if (
            medical_conditions and
            "Diabetes" in medical_conditions
        ):

            if any(
                word in disease
                for word in [
                    "infection",
                    "urinary",
                    "fungal",
                    "hepatitis"
                ]
            ):
                avg_probs[idx] *= 1.15

        # Asthma

        if (
            medical_conditions and
            "Asthma" in medical_conditions
        ):

            if any(
                word in disease
                for word in [
                    "asthma",
                    "bronchitis",
                    "pneumonia",
                    "allergy"
                ]
            ):
                avg_probs[idx] *= 1.20

        # Liver Disease

        if (
            medical_conditions and
            "Liver Disease" in medical_conditions
        ):

            if "hepatitis" in disease:
                avg_probs[idx] *= 1.25

        # Kidney Disease

        if (
            medical_conditions and
            "Kidney Disease" in medical_conditions
        ):

            if any(
                word in disease
                for word in [
                    "urinary",
                    "kidney"
                ]
            ):
                avg_probs[idx] *= 1.25

        # Heart Disease

        if (
            medical_conditions and
            "Heart Disease" in medical_conditions
        ):

            if any(
                word in disease
                for word in [
                    "heart",
                    "hypertension"
                ]
            ):
                avg_probs[idx] *= 1.20

    # Other Medical Conditions

    other_text = str(other_conditions).lower()

    if "migraine" in other_text:

        for idx, disease in enumerate(disease_names):

            if "migraine" in disease.lower():
                avg_probs[idx] *= 1.20

    if "arthritis" in other_text:

        for idx, disease in enumerate(disease_names):

            if "arthritis" in disease.lower():
                avg_probs[idx] *= 1.20

    # Normalize Probabilities
    total = avg_probs.sum()

    if total > 0:

        avg_probs = avg_probs / total

    # Top 5 Predictions

    top5 = np.argsort(
        avg_probs
    )[-5:][::-1]

    return (
        avg_probs,
        top5,
        valid,
        invalid,
        input_df
    )

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("      AI HEALTHCARE ASSISTANT")
    print("=" * 60)
    print("\nEnter symptoms separated by commas")
    print("Example: itching, skin rash, fever")

    user_input = input("\nEnter symptoms: ")
    user_symptoms = extract_symptoms(user_input, symptoms_list)
    probs, top5, valid, invalid, input_df = predict_disease(user_symptoms)
    print("\n" + "-" * 60)

    if invalid:
        print("\nIgnored / Invalid Inputs:")
        print(", ".join(invalid))

    if probs is None:
        print("\nNo valid symptoms detected.")
        exit()

    print("\nRecognized Symptoms:")
    print(", ".join(valid))

    best_idx = top5[0]
    disease = label_encoder.inverse_transform([best_idx])[0]

    print("\nMost Likely Disease:")
    print(f"{disease} ({probs[best_idx]*100:.2f}%)")

    print("\nTop 5 Predictions:")
    print("-" * 60)

    for i in top5:
        d = label_encoder.inverse_transform([i])[0]
        print(f"{d:<35} {probs[i]*100:.2f}%")

    print("-" * 60)