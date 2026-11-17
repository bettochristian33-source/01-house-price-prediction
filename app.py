import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Configuration de la page
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")

st.write(
    "Estimation du prix d'une maison "
    "avec le Machine Learning."
)

# 2. Charger les données
df = pd.read_csv("data/houses.csv")

X = df[["rooms"]]
y = df["price"]

# 3. Évaluer le modèle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2, random_state=42
)

test_model = LinearRegression()
test_model.fit(X_train, y_train)

y_pred = test_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 4. Entraîner le modèle final
model = LinearRegression()
model.fit(X, y)

# 5. Interface utilisateur
st.sidebar.header("⚙️ Paramètres")

with st.sidebar.form("prediction_form"):
    rooms = st.slider(
        "Nombre de pièces",
        min_value=1,
        max_value=7,
        value=4
    )

    st.form_submit_button("Prédire le prix")

# 6. Faire une prédiction
new_house = pd.DataFrame({"rooms": [rooms]})

prediction = model.predict(new_house)[0]

st.subheader("🎯 Résultat de la prédiction")

st.success(f"Prix estimé : {prediction:.2f}")

# 7. Afficher les performances
st.subheader("📊 Performances sur le jeu de test")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R²", f"{r2:.4f}")

# 8. Visualiser le modèle
st.subheader("📈 Régression linéaire")

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(
    df["rooms"],
    df["price"],
    color="blue",
    s=100,
    label="Données historiques"
)

x_line = pd.DataFrame({
    "rooms": np.arange(1, 8)
})

ax.plot(
    x_line["rooms"],
    model.predict(x_line),
    color="red",
    label="Modèle ML"
)

ax.scatter(
    rooms,
    prediction,
    color="green",
    s=150,
    label="Prédiction"
)

ax.set_xlabel("Nombre de pièces")
ax.set_ylabel("Prix")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# 9. Afficher les données historiques
st.subheader("📋 Dataset")

st.dataframe(df)

# 10. Afficher les limites du modèle
st.warning(
    "Modèle pédagogique entraîné sur "
    "seulement six observations."
)

st.caption("Projet 01 — Machine Learning avec Python")