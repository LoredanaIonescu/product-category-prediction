# product-category-prediction
Repozitoriu pentru dezvoltarea unui model de machine learning care prezice categoria unui produs pe baza titlului acestuia.

Ex:  
"iPhone 7 32GB Gold" → Mobile Phones

---

# Descriere proiect

Scopul proiectului este de a automatiza clasificarea produselor într-un magazin online, reducând munca manuală și erorile de categorisire.

Modelul folosește tehnici de NLP (Natural Language Processing) și Machine Learning pentru a învăța din titlurile produselor.

---

# Structura proiectului
project/
│
├── data/
│ └── products.csv
│
├── notebooks/
│ └── exploration.ipynb
│
├── src/
│ ├── train_model.py
│ └── predict_category.py
│
├── model/
│ └── model.pkl
│
└── README.md


---

# ⚙️ Tehnologii folosite

- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF Vectorization)
- Logistic Regression / Naive Bayes
- Pickle (salvare model)

---

# 🚀 Cum rulezi proiectul

## 1. Clone repository

```bash
git clone https://github.com/LoredanaIonescu/product-category-prediction.git
cd "product-category-prediction"

## 2. Antrenează modelul
python src/train_model.py

Acest script: încarcă datasetul, curăță datele, antrenează modelul, salvează modelul în model/model.pkl

## 3. Testeaza modelul
python src/predict_category.py

Introdu un titlu de produs și vei primi categoria prezisă.


## Dataset-ul conține produse dintr-un magazin online:

Product Title
Category Label
Merchant ID
Views
Rating

## Pași ML realizați
Curățare date (lowercase, eliminare simboluri)
Feature engineering (TF-IDF)
Împărțire train/test
Antrenare model ML
Evaluare model
Salvare model (.pkl)
Predicție interactivă

# Modele testate
Logistic Regression (aleasă final)
Naive Bayes

# Rezultat: Modelul prezice categoria produsului pe baza textului din titlu cu o acuratețe bună, fiind potrivit pentru automatizarea procesului de clasificare în e-commerce.