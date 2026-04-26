# ================================
# SCRIPT PENTRU ANTRENARE MODEL
# ================================

import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    """
    Curățare text:
    - lowercase
    - eliminare caractere speciale
    - spații curate
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    # ================================
    # 1. LOAD DATA
    # ================================
    df = pd.read_csv("data/products.csv")

    # eliminăm valori lipsă
    df = df.dropna(subset=["Product Title", "Category Label"])

    # curățăm textul
    df["clean_title"] = df["Product Title"].apply(clean_text)

    # ================================
    # 2. SPLIT DATE
    # ================================
    X = df["clean_title"]
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================================
    # 3. DEFINIRE MODEL
    # ================================
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # ================================
    # 4. ANTRENARE MODEL
    # ================================
    model.fit(X_train, y_train)

    # ================================
    # 5. SALVARE MODEL
    # ================================
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model antrenat și salvat cu succes!")


if __name__ == "__main__":
    main()