import pandas as pd

# Charger les données
df = pd.read_csv("data/houses.csv")

X = df["rooms"]
y = df["price"]

# Calculer les moyennes
x_mean = X.mean()
y_mean = y.mean()

# Calculer le poids
numerator = sum((X - x_mean) * (y - y_mean))
denominator = sum((X - x_mean) ** 2)

w = numerator / denominator

# Calculer le biais
b = y_mean - w * x_mean

# Construire notre modèle
def predict(x):
    return w * x + b

# Afficher les paramètres appris
print("--- MODELE ENTRAINE ---")
print(f"Weight : {w:.4f}")
print(f"Bias : {b:.4f}")

# Faire une prédiction
rooms = 4
prediction = predict(rooms)

print("\n--- PREDICTION ---")
print(f"Nombre de pièces : {rooms}")
print(f"Prix prédit : {prediction:.2f}")

# 8. Visualiser notre modèle
import matplotlib.pyplot as plt

# Calculer les prédictions
y_pred = predict(X)

# Créer le graphique
plt.figure(figsize=(10, 6))

# Afficher les données réelles
plt.scatter(
    X, y,
    color="blue",
    s=100,
    label="Données réelles"
)

# Afficher la droite de régression
plt.plot(
    X, y_pred,
    color="red",
    label="Modèle ML"
)

# Afficher notre prédiction
plt.scatter(
    rooms, prediction,
    color="green",
    s=150,
    label="Prix prédit"
)

# Personnaliser le graphique
plt.xlabel("Nombre de pièces")
plt.ylabel("Prix")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.grid(True)

# Enregistrer le graphique
plt.savefig("plots/linear_regression.png")

# Afficher le graphique
plt.show()