import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Charger les données
df = pd.read_csv("data/houses.csv")

X = df[["rooms"]]
y = df["price"]

# 2. Séparer les données
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=2,
    random_state=42
)

# 3. Créer et entraîner le modèle
model = LinearRegression()

model.fit(X_train, y_train)

# 4. Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# 5. Calculer les métriques
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 6. Afficher les dimensions
print("--- DATASET ---")
print(f"Training Set : {len(X_train)} maisons")
print(f"Test Set : {len(X_test)} maisons")

# 7. Afficher les performances
print("\n--- MODEL EVALUATION ---")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# 8. Comparer les prix réels et prédits
results = pd.DataFrame({
    "Rooms": X_test["rooms"],
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\n--- ACTUAL VS PREDICTED ---")
print(results)

