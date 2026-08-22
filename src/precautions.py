import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREC_PATH = os.path.join(BASE_DIR, "data", "raw", "symptom_precaution.csv")

try:
    precaution_df = pd.read_csv(PREC_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"❌ Precaution file not found at:\n{PREC_PATH}"
    )

def get_precautions(disease):
    data = precaution_df[precaution_df["Disease"] == disease]

    precautions = []

    if not data.empty:
        for i in range(1, 5):
            col = f"Precaution_{i}"

            if col in data.columns:
                value = data[col].values[0]

                if pd.notna(value):
                    precautions.append(value)

    return precautions