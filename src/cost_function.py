import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("data/houses.csv")

X = df["rooms"]
y = df["price"]

# Fonction de prédiction
def predict(x, w, b):
    return w * x + b

# Fonction de coût (MSE)
def cost_function(X, y, w, b):
    y_pred = predict(X, w, b)
    errors = y - y_pred
    squared_errors = errors ** 2

    return np.mean(squared_errors)

# Tester plusieurs modèles
print("--- COMPARAISON DES MODELES ---")

print("Modèle 1 :", cost_function(X, y, 0, 0))

print("Modèle 2 :", cost_function(X, y, 30, 50))

print("Modèle 3 :", cost_function(X, y, 50, 100))

print("Modèle 4 :", cost_function(X, y, 50.3929, 99.5952))