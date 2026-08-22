import os
import re
import pandas as pd
import spacy

# LOAD SPACY MODEL

nlp = spacy.load("en_core_web_sm")

# BASE DIRECTORY

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

synonyms_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "symptom_synonyms.csv"
)

if not os.path.exists(synonyms_path):
    raise FileNotFoundError(
        f"File not found:\n{synonyms_path}"
    )

synonyms_df = pd.read_csv(synonyms_path)
synonyms_df.columns = (
    synonyms_df.columns
    .str.strip()
    .str.lower()
)

# Remove empty rows

synonyms_df = synonyms_df.dropna()

# Remove duplicate rows

synonyms_df = synonyms_df.drop_duplicates()

synonyms_df["synonym"] = (
    synonyms_df["synonym"]
    .astype(str)
    .str.lower()
    .str.strip()
)

synonyms_df["dataset_symptom"] = (
    synonyms_df["dataset_symptom"]
    .astype(str)
    .str.strip()
)

SYMPTOM_MAP = dict(
    zip(
        synonyms_df["synonym"],
        synonyms_df["dataset_symptom"]
    )
)

# CLEAN TEXT

def clean_text(text):

    text = text.lower()
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()

# EXTRACT SYMPTOMS

def extract_symptoms(
    user_input,
    feature_list
):

    text = clean_text(user_input)
    doc = nlp(text)
    extracted = set()

    for synonym, symptom in SYMPTOM_MAP.items():

        if synonym in text:

            if symptom in feature_list:

                extracted.add(symptom)

    # Lemmatized Token Matching
 
    for token in doc:

        if token.is_stop or token.is_punct:
            continue

        lemma = token.lemma_.lower()

        if lemma in SYMPTOM_MAP:

            mapped = SYMPTOM_MAP[lemma]

            if mapped in feature_list:

                extracted.add(mapped)

    for symptom in feature_list:
        readable = symptom.replace("_", " ")
        if readable in text:
            extracted.add(symptom)
    return sorted(extracted)