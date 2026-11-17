import pandas as pd

# Charger le dataset
df = pd.read_csv("data/houses.csv")

# Afficher les données
print("\n--- DATASET ---")
print(df)

# Dimensions du dataset
print("\n--- DIMENSIONS ---")
print(df.shape)

# Informations sur les données
print("\n--- INFORMATIONS ---")
df.info()

# Statistiques descriptives
print("\n--- STATISTIQUES ---")
print(df.describe())

# Vérifier les valeurs manquantes
print("\n--- VALEURS MANQUANTES ---")
print(df.isnull().sum())

# Corrélation entre les variables
print("\n--- CORRELATION ---")
print(df.corr())