# ================================
# SCRIPT PENTRU PREDICȚIE
# ================================

import pickle
import re


def clean_text(text):
    """
    Curățare text identică cu cea din training
    (important pentru consistență)
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    # ================================
    # 1. ÎNCĂRCARE MODEL
    # ================================
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Model încărcat. Scrie 'exit' pentru ieșire.\n")

    # ================================
    # 2. INPUT USER + PREDICȚIE
    # ================================
    while True:
        user_input = input("Introdu titlul produsului: ")

        if user_input.lower() == "exit":
            break

        # curățăm inputul
        clean_input = clean_text(user_input)

        # facem predicția
        prediction = model.predict([clean_input])[0]

        print("Categoria prezisă:", prediction)
        print("-" * 40)


if __name__ == "__main__":
    main()