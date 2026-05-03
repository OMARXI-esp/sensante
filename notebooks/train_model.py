"""
SenSante - Lab 2 : Entraîner et Sérialiser un Modèle
"""

import pandas as pd
import numpy as np

# ===== ETAPE 2.1 : CHARGER LE DATASET =====
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# ===== ETAPE 2.2 : PREPARER LES FEATURES =====
from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"])
df["region_encoded"] = le_region.fit_transform(df["region"])

feature_cols = [
    "age",
    "sexe_encoded",
    "temperature",
    "tension_sys",
    "toux",
    "fatigue",
    "maux_tete",
    "region_encoded"
]

X = df[feature_cols]
y = df["diagnostic"]

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

# ===== ETAPE 3 : TRAIN / TEST =====
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ===== ETAPE 4 : ENTRAINEMENT =====
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# ===== ETAPE 5.1 : PREDICTION =====
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Vrai diagnostic": y_test.values[:10],
    "Prediction": y_pred[:10]
})

print(comparison)

# ===== ETAPE 5.2 : ACCURACY =====
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# ===== ETAPE 5.3 : MATRICE DE CONFUSION =====
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ===== ETAPE 5.4 : VISUALISATION =====
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Prediction du modele")
plt.ylabel("Vrai diagnostic")
plt.title("Matrice de confusion - SenSante")

plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.show()

print("Figure sauvegardee dans figures/confusion_matrix.png")

# ===== ETAPE 6 : SERIALISATION =====
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")