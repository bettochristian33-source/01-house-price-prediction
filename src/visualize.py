import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("data/houses.csv")

# Définir les variables
X = df["rooms"]
y = df["price"]

# Créer le graphique
plt.figure(figsize=(8, 6))

plt.scatter(X, y, color="blue", s=100)

# Personnaliser le graphique
plt.title("House Price Prediction")
plt.xlabel("Nombre de pièces")
plt.ylabel("Prix")
plt.grid(True)

# Enregistrer le graphique
plt.savefig("plots/houses_scatter.png")

# Afficher le graphique
plt.show()