import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Charger les données
df = pd.read_csv("data/houses.csv")

# 2. Préparer les variables
X = df[["rooms"]]
y = df["price"]

# 3. Créer le modèle
model = LinearRegression()

# 4. Entraîner le modèle
model.fit(X, y)

# 5. Afficher les paramètres appris
print("--- MODELE SCIKIT-LEARN ---")
print(f"Weight : {model.coef_[0]:.4f}")
print(f"Bias : {model.intercept_:.4f}")

# 6. Préparer une nouvelle maison
new_house = pd.DataFrame({"rooms": [4]})

# 7. Faire une prédiction
prediction = model.predict(new_house)

print("\n--- PREDICTION ---")
print("Nombre de pièces : 4")
print(f"Prix prédit : {prediction[0]:.2f}")