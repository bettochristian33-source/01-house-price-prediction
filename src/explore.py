import pandas as pd

# Création des données
data = {
    "rooms": [1, 2, 3, 5, 6, 7],
    "price": [155, 197, 244, 356, 407, 448]
}

# Création du DataFrame
df = pd.DataFrame(data)

# Affichage des données
print(df)

# Enregistrement des données
df.to_csv("data/houses.csv", index=False)

print("Dataset créé avec succès !")