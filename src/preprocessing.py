import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "Training.csv"
)

    df = pd.read_csv(train_path)
    disease_counts = df["prognosis"].value_counts()
    valid_diseases = disease_counts[disease_counts >= 2].index

    df = df[df["prognosis"].isin(valid_diseases)]
    df = df.loc[:,~df.columns.str.contains("^Unnamed")]

    # CLEAN COLUMN NAMES

    df.columns = df.columns.str.strip()
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    # FEATURES & LABEL

    if "medicine" in df.columns:

        X = df.drop(
            ["prognosis", "medicine"],
            axis=1
        )

    else:

        X = df.drop(
            ["prognosis"],
            axis=1
        )

    y = df["prognosis"]

    # LABEL ENCODING

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    models_dir = os.path.join(BASE_DIR,"models")

    os.makedirs(models_dir,exist_ok=True)

    joblib.dump(
        encoder,
        os.path.join(
            models_dir,
            "label_encoder.pkl"
        )
    )

    print("\nPreprocessing Completed")
    print(
        f"Training Shape: "
        f"{X_train.shape}"
    )

    print(
        f"Testing Shape: "
        f"{X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

if __name__ == "__main__":
    preprocess_data()