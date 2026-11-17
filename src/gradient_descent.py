import pandas as pd
import numpy as np

# 1. Charger les données
df = pd.read_csv("data/houses.csv")

X = df["rooms"].to_numpy()
y = df["price"].to_numpy()

# 2. Initialiser les paramètres
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 2000

# Historique des erreurs
loss_history = []

# 3. Entraîner le modèle
for epoch in range(epochs):

    # Calculer les prédictions
    y_pred = w * X + b

    # Calculer les erreurs
    errors = y_pred - y

    # Calculer les gradients
    dw = 2 * np.mean(X * errors)
    db = 2 * np.mean(errors)

    # Mettre à jour les paramètres
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Calculer la MSE après la mise à jour
    mse = np.mean((y - (w * X + b)) ** 2)

    loss_history.append(mse)

    # Afficher la progression
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1} | MSE : {mse:.4f}")

# 4. Afficher les paramètres appris
print("\n--- MODELE ENTRAINE ---")
print(f"Weight : {w:.4f}")
print(f"Bias : {b:.4f}")

# 5. Faire une nouvelle prédiction
rooms = 4
prediction = w * rooms + b

print("\n--- PREDICTION ---")
print(f"Nombre de pièces : {rooms}")
print(f"Prix prédit : {prediction:.2f}")

# 6. Visualiser la convergence du modèle
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    color="blue",
    label="Training Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent - Training Loss")

plt.yscale("log")

plt.legend()
plt.grid(True)

plt.savefig("plots/gradient_descent_loss.png")

plt.show()