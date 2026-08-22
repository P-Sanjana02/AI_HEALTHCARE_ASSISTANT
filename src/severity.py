import pandas as pd
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

severity_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Symptom-severity.csv"
)

severity_df = pd.read_csv(severity_path)
severity_df.columns = (severity_df.columns.str.strip())
severity_map = dict(
    zip(
        severity_df["Symptom"].str.strip().str.lower(),
        severity_df["weight"]
    )
)

# HIGH RISK SYMPTOMS

HIGH_RISK_SYMPTOMS = {

    "chest pain",
    "breathlessness",
    "fast heart rate",
    "irregular sugar level",
    "coma",
    "unconsciousness",
    "loss of balance",
    "slurred speech",
    "blood in sputum",
    "blackouts",
    "acute liver failure"
}

# HIGH RISK DISEASES

HIGH_RISK_DISEASES = {

    "Heart attack",
    "Pneumonia",
    "Malaria",
    "Dengue",
    "Typhoid",
    "Tuberculosis",
    "Stroke",
    "Hepatitis E",
    "Hepatitis D",
    "AIDS"
}

# SEVERITY FUNCTION

def get_severity(
        valid_symptoms,
        medical_conditions=None,
        predicted_disease=None
):

    total_score = 0

    # Symptom Severity Score

    for symptom in valid_symptoms:

        clean_symptom = (
            symptom
            .replace("_", " ")
            .strip()
            .lower()
        )

        total_score += severity_map.get(
            clean_symptom,
            1
        )

    # Medical Condition Weight

    if medical_conditions:

        if "Diabetes" in medical_conditions:
            total_score += 1

        if "BP" in medical_conditions:
            total_score += 1

        if "Heart Disease" in medical_conditions:
            total_score += 3

        if "Kidney Disease" in medical_conditions:
            total_score += 2

        if "Liver Disease" in medical_conditions:
            total_score += 2

        if "Asthma" in medical_conditions:
            total_score += 2

        if "Thyroid" in medical_conditions:
            total_score += 1

        if "Gastric" in medical_conditions:
            total_score += 1

    for symptom in valid_symptoms:

        clean_symptom = (
            symptom
            .replace("_", " ")
            .strip()
            .lower()
        )

        if clean_symptom in HIGH_RISK_SYMPTOMS:

            return (
                "HIGH RISK",
                10
            )

    if predicted_disease:

        if predicted_disease in HIGH_RISK_DISEASES:

            return ("HIGH RISK",10)

    final_score = min(round(total_score / 5, 2),10)

    # Risk Levels

    if final_score < 3:
        level = "LOW RISK"
    elif final_score < 6:
        level = "MEDIUM RISK"
    else:
        level = "HIGH RISK"
    return (level,final_score)