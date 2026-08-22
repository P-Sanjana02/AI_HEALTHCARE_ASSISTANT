import os
import joblib
import shap
import pandas as pd
import numpy as np

# BASE DIRECTORY

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

desc_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "symptom_Description.csv"
)

desc_df = pd.read_csv(desc_path)
desc_df.columns = desc_df.columns.str.strip()

# LOAD MODELS

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)

rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

rf_explainer = shap.TreeExplainer(rf_model)
xgb_explainer = shap.TreeExplainer(xgb_model)

# DISEASE DESCRIPTION

def explain_disease(disease):

    disease = str(disease).strip()

    row = desc_df[
        desc_df["Disease"].str.strip() == disease
    ]["Description"]

    if not row.empty:
        return row.values[0]

    return "Medical description unavailable."


# SHAP EXPLANATION

def get_shap_explanation(input_df, predicted_index):

    # RANDOM FOREST SHAP

    rf_shap_values = rf_explainer(input_df)

    if rf_shap_values.values.ndim == 3:

        rf_class_shap = (
            rf_shap_values.values[
                0,
                :,
                predicted_index
            ]
        )

    else:

        rf_class_shap = (
            rf_shap_values.values[0]
        )


    # XGBOOST SHAP

    xgb_shap_values = xgb_explainer(input_df)

    if xgb_shap_values.values.ndim == 3:

        xgb_class_shap = (
            xgb_shap_values.values[
                0,
                :,
                predicted_index
            ]
        )

    else:

        xgb_class_shap = (
            xgb_shap_values.values[0]
        )


    # COMBINE SHAP VALUES
    # Same weights as ensemble prediction

    combined_shap = (
        0.6 * rf_class_shap
        + 0.4 * xgb_class_shap
    )


    # FEATURE IMPORTANCE

    feature_importance = pd.DataFrame({

        "Symptom": input_df.columns,

        "Importance": np.abs(
            combined_shap
        )

    })


    # ONLY USER-ENTERED SYMPTOMS

    selected_feature = (
        input_df.columns[
            input_df.iloc[0] == 1
        ]
    )

    feature_importance = feature_importance[
        feature_importance["Symptom"].isin(
            selected_feature
        )
    ]


    # SORT BY IMPORTANCE

    feature_importance = (
        feature_importance.sort_values(
            by="Importance",
            ascending=False
        )
    )


    # READABLE SYMPTOM NAMES

    feature_importance["Symptom"] = (
        feature_importance["Symptom"]
        .str.replace("_", " ")
        .str.title()
    )

    feature_importance = (
        feature_importance.head(5)
    )

    return feature_importance