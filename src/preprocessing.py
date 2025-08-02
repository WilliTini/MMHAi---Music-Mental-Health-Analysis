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

# %% [markdown]
# ## Caricamento e preparazione dati

# %%
# Load the cleaned dataset
mxmh = pd.read_csv("data/mxmh_clean.csv")
print(f"Dataset caricato: {mxmh.shape}", flush=True)
print(f"Colonne: {list(mxmh.columns)}", flush=True)

# %%
# Define categories for listener hours
# This function categorizes the hours of listening into different categories
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
print("Categoria di ascolto creata", flush=True)

# %% 
# Grafico distribuzione età
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Age"], kde=True, bins=40)
plt.title("Distribution of Age")
plt.show()
print("Grafico età mostrato")

# %%
# Grafico distribuzione categoria ascolto
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Listening Category"], kde=False, bins=30)
plt.title("Distribution of Listening Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Grafico categoria ascolto mostrato")

# %%
# Grafico distribuzione servizi streaming
plt.figure(figsize=(12, 6))
sns.histplot(mxmh["Primary streaming service"], kde=False, bins=30)
plt.title("Distribution of Primary Streaming Service")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Grafico servizi streaming mostrato")

# %%
# Grafico distribuzione ore per giorno
plt.figure(figsize=(10, 6))
sns.histplot(mxmh["Hours per day"], kde=False, bins=30)
plt.title("Distribution of Hours per Day")
plt.show()

# %%
# Grafici salute mentale
mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
for col in mental_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(mxmh[col], kde=True, bins=20)
    plt.title(f"Distribuzione di {col}")
    plt.show()

# %%
# Matrice di correlazione con Hours per day
mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
correlation = mxmh[mental_cols + ["Hours per day"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Matrice di correlazione con Hours per day")
plt.show()

# %%
# Analisi salute mentale per categoria di listener
# Creiamo grafici separati per ogni categoria di ascolto
mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
listener_categories = ["Very Low", "Low", "Mean", "High", "Very High"]

for category in listener_categories:
    # Filtriamo i dati per la categoria specifica
    category_data = mxmh[mxmh["Listening Category"] == category]
    
    if len(category_data) > 0:  # Controlliamo che ci siano dati per questa categoria
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Distribuzione Salute Mentale - Listener {category} (n={len(category_data)})', fontsize=16)
        
        for i, col in enumerate(mental_cols):
            row, col_idx = i // 2, i % 2
            axes[row, col_idx].hist(category_data[col], bins=10, alpha=0.7, edgecolor='black')
            axes[row, col_idx].set_title(f'{col}')
            axes[row, col_idx].set_xlabel('Score')
            axes[row, col_idx].set_ylabel('Frequenza')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Nessun dato trovato per la categoria {category}")
        
print("Grafici per categoria completati!")

# %%
# Grafico di confronto delle medie per categoria
# Calcoliamo le medie per ogni categoria e condizione
mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
listener_categories = ["Very Low", "Low", "Mean", "High", "Very High"]

# Creiamo un DataFrame con le medie
means_data = []
for category in listener_categories:
    category_data = mxmh[mxmh["Listening Category"] == category]
    if len(category_data) > 0:
        for col in mental_cols:
            means_data.append({
                'Listener_Category': category,
                'Mental_Health_Condition': col,
                'Mean_Score': category_data[col].mean(),
                'Count': len(category_data)
            })

means_df = pd.DataFrame(means_data)

# Creiamo un grafico a barre raggruppate
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(listener_categories))
width = 0.2

for i, condition in enumerate(mental_cols):
    condition_data = means_df[means_df['Mental_Health_Condition'] == condition]
    scores = [condition_data[condition_data['Listener_Category'] == cat]['Mean_Score'].iloc[0] 
              if len(condition_data[condition_data['Listener_Category'] == cat]) > 0 else 0 
              for cat in listener_categories]
    
    ax.bar(x + i*width, scores, width, label=condition)

ax.set_xlabel('Categoria di Ascolto')
ax.set_ylabel('Score Medio')
ax.set_title('Confronto Score Medi di Salute Mentale per Categoria di Listener')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(listener_categories, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()


