#Analisi Dati
# This script performs data analysis on the MMH survey results dataset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

 
# Load the dataset
#mxmh = pd.read_csv("../data/mxmh_survey_results.csv")

# Clean data from useless and ridodant columns
# and check for missing values
# mxmh.isnull().sum()
# mxmh = mxmh.drop(columns=["Timestamp", "Permissions", "BPM", "Music effects"])
# mxmh["Age"] = mxmh["Age"].fillna(mxmh["Age"].mean())
# mxmh["Primary streaming service"] = mxmh["Primary streaming service"].fillna("Spotify")

# Save the cleaned dataset
#mxmh.to_csv("mxmh_clean.csv", index=False)

# Load the cleaned dataset
mxmh = pd.read_csv("data/mxmh_clean.csv")
print(f"Dataset caricato: {mxmh.shape}")
print(f"Colonne: {list(mxmh.columns)}")


def listener_categories(h):
    if h < 1:
        return "Very Low"
    elif 1 <= h < 2:
        return "Low"
    elif 2 <= h < 4.5:
        return "Mean"
    elif 4.5 <= h < 6.5:
        return "High"
    elif h >= 6:
        return "Very High"
    
mxmh["Listening Category"] = mxmh["Hours per day"].apply(listener_categories)
print("Categoria di ascolto creata")

    
# Plotting the distribution of age
print("Mostrando grafico età...")
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Age"], kde=True, bins=40)
plt.title("Distribution of Age")
plt.show()
print("Grafico età mostrato")

# Plotting the distribution of listening category
print("Mostrando grafico categoria ascolto...")
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Listening Category"], kde=False, bins=30)
plt.title("Distribution of Listening Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Grafico categoria ascolto mostrato")

# Plotting the distribution of primary streaming service
print("Mostrando grafico servizi streaming...")
plt.figure(figsize=(12, 6))
sns.histplot(mxmh["Primary streaming service"], kde=False, bins=30)
plt.title("Distribution of Primary Streaming Service")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Grafico servizi streaming mostrato")

# Plotting the distribution of Hours per day
print("Mostrando grafico ore per giorno...")
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Hours per day"], kde=False, bins=30)
plt.title("Distribution of Hours per Day")
plt.show()
print("Grafico ore per giorno mostrato")

# Plotting the distribution of Listening Category
print("Mostrando grafico categoria ascolto...")
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Listening Category"], kde=False, bins=30)
plt.title("Distribution of Listening Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Grafico categoria ascolto mostrato")

print("Analisi completata! I grafici sono stati mostrati in finestre separate.")