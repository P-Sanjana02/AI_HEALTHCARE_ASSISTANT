import os
import joblib
import pandas as pd

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from preprocessing import preprocess_data

# PREPROCESS DATA

X_train, X_test, y_train, y_test = preprocess_data()
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# RANDOM FOREST

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)
rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="weighted",
    zero_division=0
)

rf_cv = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5
)

print("\n RANDOM FOREST ")
print(f"Accuracy              : {rf_accuracy:.4f}")
print(f"Precision             : {rf_precision:.4f}")
print(f"Recall                : {rf_recall:.4f}")
print(f"F1 Score              : {rf_f1:.4f}")
print(f"Cross Validation Mean : {rf_cv.mean():.4f}")
print("\nClassification Report")
print(classification_report(y_test, rf_predictions, zero_division=0))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,rf_predictions))

# XGBOOST

print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=2,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

xgb_precision = precision_score(
    y_test,
    xgb_predictions,
    average="weighted",
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_predictions,
    average="weighted",
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_predictions,
    average="weighted",
    zero_division=0
)

xgb_cv = cross_val_score(
    xgb_model,
    X_train,
    y_train,
    cv=5
)

print("\n XGBOOST ")
print(f"Accuracy              : {xgb_accuracy:.4f}")
print(f"Precision             : {xgb_precision:.4f}")
print(f"Recall                : {xgb_recall:.4f}")
print(f"F1 Score              : {xgb_f1:.4f}")
print(f"Cross Validation Mean : {xgb_cv.mean():.4f}")
print("\nClassification Report")
print(
    classification_report(
        y_test,
        xgb_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        xgb_predictions
    )
)

# MODEL COMPARISON

comparison = pd.DataFrame({

    "Model": [
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],

    "Precision": [
        rf_precision,
        xgb_precision
    ],

    "Recall": [
        rf_recall,
        xgb_recall
    ],

    "F1 Score": [
        rf_f1,
        xgb_f1
    ],

    "Cross Validation": [
        rf_cv.mean(),
        xgb_cv.mean()
    ]

})

print("\n MODEL COMPARISON \n")

print(comparison)

comparison.to_csv(

    os.path.join(
        models_dir,
        "model_comparison.csv"
    ),

    index=False
)

joblib.dump(rf_model,os.path.join(models_dir, "random_forest_model.pkl"))
joblib.dump(xgb_model,os.path.join(models_dir,"xgboost_model.pkl"))
print("\nModels saved successfully!")
print("Model comparison saved as model_comparison.csv")