#Analisi Dati
# This script performs data analysis on the MMH survey results dataset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

 
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
# ## Caricamento e preparazione dati

# %%
# Load the cleaned dataset
mxmh = pd.read_csv("data/mxmh_final.csv")
# Gestione dei valori mancanti per BPM
if 'BPM' in mxmh.columns:
    mxmh['BPM'] = mxmh['BPM'].fillna(mxmh['BPM'].mean())
    print("Valori mancanti in 'BPM' riempiti con la media.")

print(f"Dataset caricato: {mxmh.shape}", flush=True)
print(f"Colonne: {list(mxmh.columns)}", flush=True)

# %%
# Distribuzione delle feature numeriche principali
print("Distribuzione delle feature numeriche principali")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribuzione delle Feature Numeriche Principali')

sns.histplot(mxmh['Age'], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribuzione Età')

sns.histplot(mxmh['Hours per day'], bins=20, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribuzione Ore di Ascolto Giornaliere')

sns.histplot(mxmh['Depression'], bins=10, kde=False, ax=axes[1, 0])
axes[1, 0].set_title('Distribuzione Score Depressione')

sns.histplot(mxmh['Anxiety'], bins=10, kde=False, ax=axes[1, 1])
axes[1, 1].set_title('Distribuzione Score Ansia')

plt.tight_layout()
plt.show()

# Box plot per feature categoriche vs ore di ascolto
print("Box plot per feature categoriche vs ore di ascolto")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Ore di Ascolto vs Feature Categoriche')

sns.boxplot(data=mxmh, x='Instrumentalist', y='Hours per day', ax=axes[0, 0])
axes[0, 0].set_title('Ore di Ascolto vs Essere Strumentista')

sns.boxplot(data=mxmh, x='Composer', y='Hours per day', ax=axes[0, 1])
axes[0, 1].set_title('Ore di Ascolto vs Essere Compositore')

sns.boxplot(data=mxmh, x='Foreign languages', y='Hours per day', ax=axes[1, 0])
axes[1, 0].set_title('Ore di Ascolto vs Ascolto Lingue Straniere')

sns.boxplot(data=mxmh, x='While working', y='Hours per day', ax=axes[1, 1])
axes[1, 1].set_title('Ore di Ascolto vs Ascolto Durante il Lavoro')

plt.tight_layout()
plt.show()

# Box plot per Salute Mentale vs Strumentista/Compositore
print("Box plot per Salute Mentale vs Strumentista/Compositore")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Salute Mentale vs Attività Musicali')

sns.boxplot(data=mxmh, x='Instrumentalist', y='Depression', ax=axes[0, 0])
axes[0, 0].set_title('Depressione vs Essere Strumentista')

sns.boxplot(data=mxmh, x='Instrumentalist', y='Anxiety', ax=axes[0, 1])
axes[0, 1].set_title('Ansia vs Essere Strumentista')

sns.boxplot(data=mxmh, x='Composer', y='Depression', ax=axes[1, 0])
axes[1, 0].set_title('Depressione vs Essere Compositore')

sns.boxplot(data=mxmh, x='Composer', y='Anxiety', ax=axes[1, 1])
axes[1, 1].set_title('Ansia vs Essere Compositore')

plt.tight_layout()
plt.show()

# %%
# Applicazione della categorizzazione basata sui dati effettivi
def listener_categories(h):
    """
    Categorizzazione basata sui percentili del dataset:
    - Very Low: < 2 ore (sotto il 25° percentile)
    - Low: 2-3 ore (25°-50° percentile) 
    - Mean: 3-5 ore (50°-75° percentile)
    - High: 5-7 ore (75°-90° percentile)
    - Very High: >= 7 ore (sopra il 90° percentile)
    """
    if h < 2:
        return "Very Low"
    elif 2 <= h < 3:
        return "Low"
    elif 3 <= h < 5:
        return "Mean"
    elif 5 <= h < 7:
        return "High"
    else:  # h >= 7
        return "Very High"
    
mxmh["Listening Category"] = mxmh["Hours per day"].apply(listener_categories)
print("Categoria di ascolto RIVISTA creata", flush=True)

# Verifica la distribuzione
new_distribution = mxmh["Listening Category"].value_counts()
print("\nDistribuzione:")
for cat, count in new_distribution.items():
    percentage = (count / len(mxmh) * 100)
    print(f"- {cat}: {count} persone ({percentage:.1f}%)")

# Visualizza la distribuzione
plt.figure(figsize=(10, 6))
bars = plt.bar(new_distribution.index, new_distribution.values, alpha=0.7, color='lightblue')
plt.title('Distribuzione delle Categorie di Ascolto')
plt.xlabel('Categoria')
plt.ylabel('Numero Persone')
plt.xticks(rotation=45)

# Aggiunta le percentuali sopra le barre
for i, (cat, count) in enumerate(new_distribution.items()):
    percentage = (count / len(mxmh) * 100)
    plt.text(i, count + 5, f'{percentage:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n=== NUOVA CATEGORIZZAZIONE ===")
print("RIVISTA (basata sui dati):")
print("- Very Low: < 2 ore (sotto 25° percentile)")
print("- Low: 2-3 ore (25°-50° percentile)")
print("- Mean: 3-5 ore (50°-75° percentile)")
print("- High: 5-7 ore (75°-90° percentile)")
print("- Very High: >= 7 ore (sopra 90° percentile)")

print(f"\nLa nuova categorizzazione è più bilanciata e riflette meglio")
print(f"la distribuzione effettiva del dataset (mediana: 3 ore, media: {mxmh['Hours per day'].mean():.1f} ore)")


# %%
# Analisi salute mentale per categoria di listener
mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
listener_categories_list = ["Very Low", "Low", "Mean", "High", "Very High"]

# Creazione di un DataFrame con le medie delle categorie
means_data = []
for category in listener_categories_list:
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

# Creazione di un grafico a barre raggruppate
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(listener_categories_list))
width = 0.2

for i, condition in enumerate(mental_cols):
    condition_data = means_df[means_df['Mental_Health_Condition'] == condition]
    scores = [condition_data[condition_data['Listener_Category'] == cat]['Mean_Score'].iloc[0] 
              if len(condition_data[condition_data['Listener_Category'] == cat]) > 0 else 0 
              for cat in listener_categories_list]
    
    ax.bar(x + i*width, scores, width, label=condition)

ax.set_xlabel('Categoria di Ascolto')
ax.set_ylabel('Score Medio')
ax.set_title('Confronto Score Medi di Salute Mentale per Categoria di Listener (RIVISTA)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(listener_categories_list, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()


# %%
# Clustering con categorie di listener RIVISTE
listener_mapping = {
    "Very Low": 0,
    "Low": 1,
    "Mean": 2,
    "High": 3,
    "Very High": 4
}
mxmh["Listener_Type_Num"] = mxmh["Listening Category"].map(listener_mapping)

features = mxmh[["Listener_Type_Num", "Depression", "Anxiety", "Insomnia", "OCD"]]
features_clean = features.dropna()

scaler = StandardScaler()
X = scaler.fit_transform(features_clean)

# Test per trovare il numero ottimale di cluster
print("=== CLUSTERING CON NUOVA CATEGORIZZAZIONE ===")
silhouette_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nMigliore k: {best_k} (silhouette: {max(silhouette_scores):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = km_final.fit_predict(X)

pca = PCA(2)
X_pca = pca.fit_transform(X)

# Visualizzazione PCA
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:,0], X_pca[:,1], c=final_labels, cmap="tab10", alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'Clustering K-means (k={best_k}) - Nuova Categorizzazione')
plt.colorbar(label='Cluster')
plt.show()

# Analisi dei cluster
mxmh_with_clusters = mxmh.loc[features_clean.index].copy()
mxmh_with_clusters['Cluster'] = final_labels

print(f"\nDimensioni cluster:")
print(mxmh_with_clusters['Cluster'].value_counts().sort_index())

print("\n=== ANALISI CLUSTER CON NUOVA CATEGORIZZAZIONE ===")
for cluster_id in sorted(mxmh_with_clusters['Cluster'].unique()):
    cluster_data = mxmh_with_clusters[mxmh_with_clusters['Cluster'] == cluster_id]
    print(f"\nCLUSTER {cluster_id} (n={len(cluster_data)} persone)")
    
    # Percentuali delle categorie di listener
    listener_counts = cluster_data['Listening Category'].value_counts()
    listener_percentages = (listener_counts / len(cluster_data) * 100).round(1)
    
    print("Distribuzione categorie ascolto:")
    for category, percentage in listener_percentages.items():
        print(f"   {category}: {listener_counts[category]} persone ({percentage}%)")
    
    # Medie salute mentale
    mental_means = cluster_data[["Depression", "Anxiety", "Insomnia", "OCD"]].mean()
    print("Medie salute mentale:")
    for condition, mean_val in mental_means.items():
        print(f"   {condition}: {mean_val:.2f}")

# %%
# Grafico distribuzione listener per cluster
fig, axes = plt.subplots(1, best_k, figsize=(5*best_k, 6))
if best_k == 1:
    axes = [axes]

for cluster_id in range(best_k):
    cluster_data = mxmh_with_clusters[mxmh_with_clusters['Cluster'] == cluster_id]
    listener_counts = cluster_data['Listening Category'].value_counts()
    
    category_order = ["Very Low", "Low", "Mean", "High", "Very High"]
    ordered_counts = [listener_counts.get(cat, 0) for cat in category_order]
    
    bars = axes[cluster_id].bar(category_order, ordered_counts, 
                               color=plt.cm.tab10(cluster_id), alpha=0.7)
    axes[cluster_id].set_title(f'Cluster {cluster_id}\n(n={len(cluster_data)})')
    axes[cluster_id].set_xlabel('Categoria Ascolto')
    axes[cluster_id].set_ylabel('Numero Persone')
    axes[cluster_id].tick_params(axis='x', rotation=45)
    
    # Aggiunta i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[cluster_id].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                 f'{int(height)}', ha='center', va='bottom')

plt.suptitle('Distribuzione Categorie Listener per Cluster (NUOVA CATEGORIZZAZIONE)', fontsize=16)
plt.tight_layout()
plt.show()

# Visualizzazione delle medie salute mentale per cluster
cluster_analysis = mxmh_with_clusters.groupby("Cluster")[["Depression", "Anxiety", "Insomnia", "OCD", "Listener_Type_Num"]].mean()

mental_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Confronto Medie Salute Mentale per Cluster (NUOVA CATEGORIZZAZIONE)', fontsize=16)

for i, col in enumerate(mental_cols):
    row, col_idx = i // 2, i % 2
    cluster_means = cluster_analysis[col]
    
    bars = axes[row, col_idx].bar(range(len(cluster_means)), cluster_means.values, 
                                  color=plt.cm.tab10(range(len(cluster_means))), alpha=0.7)
    axes[row, col_idx].set_title(f'{col}')
    axes[row, col_idx].set_xlabel('Cluster')
    axes[row, col_idx].set_ylabel('Score Medio')
    axes[row, col_idx].set_xticks(range(len(cluster_means)))
    axes[row, col_idx].set_xticklabels([f'Cluster {i}' for i in cluster_means.index])
    
    # Aggiunta i valori sopra le barre
    for j, bar in enumerate(bars):
        height = bar.get_height()
        axes[row, col_idx].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# ANALISI INTERPRETATIVA APPROFONDITA
print("\n" + "="*80)
print("ANALISI INTERPRETATIVA APPROFONDITA DEI CLUSTER")
print("="*80)

# Calcoliamo statistiche dettagliate per ogni cluster
for cluster_id in sorted(mxmh_with_clusters['Cluster'].unique()):
    cluster_data = mxmh_with_clusters[mxmh_with_clusters['Cluster'] == cluster_id]

    print(f"\nCLUSTER {cluster_id} - PROFILO DETTAGLIATO")
    print(f"{'='*60}")
    print(f"Dimensione: {len(cluster_data)} persone ({len(cluster_data)/len(mxmh_with_clusters)*100:.1f}% del campione)")
    
    # Analisi ore di ascolto
    hours_stats = cluster_data['Hours per day'].describe()
    print(f"\nPATTERN DI ASCOLTO:")
    print(f"   • Media ore/giorno: {hours_stats['mean']:.2f}")
    print(f"   • Mediana: {hours_stats['50%']:.2f}")
    print(f"   • Range: {hours_stats['min']:.1f} - {hours_stats['max']:.1f} ore")
    print(f"   • Deviazione standard: {hours_stats['std']:.2f}")
    
    # Distribuzione categorica dettagliata
    listener_dist = cluster_data['Listening Category'].value_counts(normalize=True) * 100
    print(f"\nDISTRIBUZIONE CATEGORIE:")
    category_order = ["Very Low", "Low", "Mean", "High", "Very High"]
    for cat in category_order:
        if cat in listener_dist:
            count = cluster_data['Listening Category'].value_counts()[cat]
            print(f"   • {cat:10}: {count:3d} persone ({listener_dist[cat]:5.1f}%)")
    
    # Analisi salute mentale dettagliata
    mental_stats = cluster_data[mental_cols].describe()
    print(f"\nPROFILO SALUTE MENTALE:")
    for condition in mental_cols:
        mean_val = mental_stats.loc['mean', condition]
        std_val = mental_stats.loc['std', condition]
        min_val = mental_stats.loc['min', condition]
        max_val = mental_stats.loc['max', condition]
        print(f"   • {condition:10}: μ={mean_val:4.2f} ± {std_val:4.2f} (range: {min_val:.0f}-{max_val:.0f})")
    
    # Correlazioni interne al cluster
    cluster_corr = cluster_data[['Hours per day'] + mental_cols].corr()['Hours per day'].drop('Hours per day')
    print(f"\nCORRELAZIONI ORE-ASCOLTO vs SALUTE MENTALE:")
    for condition in mental_cols:
        corr_val = cluster_corr[condition]
        strength = "forte" if abs(corr_val) > 0.3 else "moderata" if abs(corr_val) > 0.1 else "debole"
        direction = "positiva" if corr_val > 0 else "negativa"
        print(f"   • {condition:10}: r={corr_val:+5.2f} ({strength} {direction})")

# %%
# INTERPRETAZIONE TEORICA
print(f"\n" + "="*80)
print(f"INTERPRETAZIONE TEORICA E IMPLICAZIONI")
print("="*80)

# Calcoliamo le caratteristiche distintive di ogni cluster
cluster_profiles = {}
for cluster_id in sorted(mxmh_with_clusters['Cluster'].unique()):
    cluster_data = mxmh_with_clusters[mxmh_with_clusters['Cluster'] == cluster_id]
    
    profile = {
        'size': len(cluster_data),
        'avg_hours': cluster_data['Hours per day'].mean(),
        'avg_depression': cluster_data['Depression'].mean(),
        'avg_anxiety': cluster_data['Anxiety'].mean(),
        'avg_insomnia': cluster_data['Insomnia'].mean(),
        'avg_ocd': cluster_data['OCD'].mean(),
        'dominant_category': cluster_data['Listening Category'].mode().iloc[0],
        'mental_health_avg': cluster_data[mental_cols].mean().mean()
    }
    cluster_profiles[cluster_id] = profile

print("\nPROFILI EMERSI:")
for cluster_id, profile in cluster_profiles.items():
    print(f"\nCLUSTER {cluster_id}:")
    print(f"   • Ascolto medio: {profile['avg_hours']:.1f} ore/giorno")
    print(f"   • Categoria dominante: {profile['dominant_category']}")
    print(f"   • Score salute mentale medio: {profile['mental_health_avg']:.2f}")
    
    # Classificazione del cluster
    if profile['mental_health_avg'] < 4:
        health_level = "BASSA severità"
    elif profile['mental_health_avg'] < 6:
        health_level = "MEDIA severità"  
    else:
        health_level = "ALTA severità"

    print(f"Livello salute mentale: {health_level}")

# %%
# IPOTESI INTERPRETATIVE
print(f"\nIPOTESI INTERPRETATIVE:")

# Identifichiamo il cluster con problemi maggiori e minori
high_distress_cluster = max(cluster_profiles.keys(), key=lambda x: cluster_profiles[x]['mental_health_avg'])
low_distress_cluster = min(cluster_profiles.keys(), key=lambda x: cluster_profiles[x]['mental_health_avg'])

print(f"\n1. CLUSTER {low_distress_cluster} - 'ASCOLTATORI SALUTARI':")
print(f"   • Caratterizzato da bassi livelli di problematiche psicologiche")
print(f"   • Pattern di ascolto: {cluster_profiles[low_distress_cluster]['avg_hours']:.1f} ore in media")
print(f"   • IPOTESI: Uso della musica come risorsa positiva per il benessere")
print(f"   • La musica potrebbe avere funzione rilassante/regolativa")

print(f"\n2. CLUSTER {high_distress_cluster} - 'ASCOLTATORI IN DIFFICOLTÀ':")
print(f"   • Elevati livelli di sintomi psicologici")
print(f"   • Pattern di ascolto variegato: {cluster_profiles[high_distress_cluster]['avg_hours']:.1f} ore in media")
print(f"   • IPOTESI: Possibile uso della musica come coping maladattivo")
print(f"   • Oppure la musica come tentativo di autoregolazione emotiva")

# Correlazioni globali
overall_corr = mxmh_with_clusters[['Hours per day'] + mental_cols].corr()['Hours per day'].drop('Hours per day')
print(f"\n3. RELAZIONE MUSICA-SALUTE MENTALE:")
print(f"   • Correlazioni ore di ascolto con problematiche psicologiche:")
for condition in mental_cols:
    corr_val = overall_corr[condition]
    if abs(corr_val) > 0.1:
        direction = "Maggiore" if corr_val > 0 else "Minore"
        print(f"     - {direction} ascolto → {direction.lower()} {condition} (r={corr_val:+.3f})")

print(f"\nCONCLUSIONI CHIAVE:")
print(f"   ✓ Emergono profili distinti di relazione musica-benessere psicologico")
print(f"   ✓ La categorizzazione rivista permette di identificare pattern più chiari")
print(f"   ✓ Necessario approfondire i meccanismi causali (musica come causa/effetto/mediatore)")

print("\nAnalisi interpretativa completata!")

# %%
# CLUSTERING COMPLESSIVO SUI PROFILI D'ASCOLTO
print("\n" + "="*80)
print("CLUSTERING COMPLESSIVO SUI PROFILI D'ASCOLTO")
print("="*80)

# 1. PREPARAZIONE DATI
print("\n1. PREPARAZIONE DATI PER CLUSTERING COMPLESSIVO...")

# Mapping delle frequenze in valori numerici (se non già fatto)
frequency_mapping = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Very frequently': 3
}
genre_cols = [col for col in mxmh.columns if 'Frequency' in col]
for col in genre_cols:
    if f"{col}_numeric" not in mxmh.columns:
        mxmh[f"{col}_numeric"] = mxmh[col].map(frequency_mapping)

# Conversione variabili binarie (se non già fatto)
if 'Instrumentalist_numeric' not in mxmh.columns:
    mxmh['Instrumentalist_numeric'] = (mxmh['Instrumentalist'] == 'Yes').astype(int)
if 'Composer_numeric' not in mxmh.columns:
    mxmh['Composer_numeric'] = (mxmh['Composer'] == 'Yes').astype(int)

# Definizione delle feature per il clustering dei profili musicali
genre_numeric_cols = [f"{col}_numeric" for col in genre_cols]
profile_features_cols = [
    'Age', 
    'Hours per day',
    'BPM',
    'Instrumentalist_numeric',
    'Composer_numeric'
] + genre_numeric_cols

# Controllo per valori mancanti residui (es. in colonne genere)
for col in profile_features_cols:
    if mxmh[col].isnull().any():
        # Riempiamo eventuali NaN residui con 0. Questo è sensato per le 
        # frequenze dei generi (NaN -> non ascoltato) e per altre feature
        # dove un valore mancante può essere considerato uno zero.
        mxmh[col] = mxmh[col].fillna(0)
        print(f"Valori mancanti nella colonna '{col}' riempiti con 0.")


profile_features = mxmh[profile_features_cols].copy() # Usare .copy() per evitare warning

# Riduci l'impatto dell'età applicando un peso
age_weight = 0.1
profile_features['Age'] = profile_features['Age'] * age_weight
print(f"\nApplicato un peso di {age_weight} alla feature 'Age' per ridurne l'impatto.")

print(f"\nFeature utilizzate per il clustering ({len(profile_features_cols)}):")
print(profile_features_cols)
print(f"\nCampione per il clustering: {len(profile_features)} persone")

# 2. STANDARDIZZAZIONE E CLUSTERING
print("\n2. STANDARDIZZAZIONE E RICERCA K OTTIMALE...")
scaler_profile = StandardScaler()
X_profile = scaler_profile.fit_transform(profile_features)

# Troviamo il numero ottimale di cluster
silhouette_scores_profile = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_profile)
    score = silhouette_score(X_profile, labels)
    silhouette_scores_profile.append(score)
    print(f"k={k}, silhouette={score:.3f}")

best_k_profile = k_range[silhouette_scores_profile.index(max(silhouette_scores_profile))]
print(f"\nMigliore k: {best_k_profile} (silhouette: {max(silhouette_scores_profile):.3f})")

# Clustering finale
km_profile = KMeans(n_clusters=best_k_profile, random_state=42, n_init=10)
profile_labels = km_profile.fit_predict(X_profile)

# Aggiungiamo i cluster al dataframe principale
mxmh['Profile_Cluster'] = profile_labels

# 3. VISUALIZZAZIONE CLUSTER (PCA)
print("\n3. VISUALIZZAZIONE CLUSTER TRAMITE PCA...")
pca_profile = PCA(n_components=2)
X_profile_pca = pca_profile.fit_transform(X_profile)

plt.figure(figsize=(12, 9))
sns.scatterplot(x=X_profile_pca[:, 0], y=X_profile_pca[:, 1], hue=profile_labels, 
                palette='viridis', alpha=0.8, s=50)
plt.xlabel(f'Componente Principale 1 ({pca_profile.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'Componente Principale 2 ({pca_profile.explained_variance_ratio_[1]:.2%} varianza)')
plt.title(f'Clustering dei Profili Musicali (k={best_k_profile}) - Vista PCA')
plt.legend(title='Cluster')
plt.show()

# 4. ANALISI E INTERPRETAZIONE DEI PROFILI CLUSTER
print("\n" + "="*80)
print("4. ANALISI E INTERPRETAZIONE DEI PROFILI CLUSTER")
print("="*80)

# Aggiungiamo anche le colonne sulla salute mentale per l'analisi finale
analysis_cols = profile_features_cols + ['Depression', 'Anxiety', 'Insomnia', 'OCD', 'Profile_Cluster']
mxmh_profiles = mxmh[analysis_cols]

for cluster_id in sorted(mxmh_profiles['Profile_Cluster'].unique()):
    cluster_data = mxmh_profiles[mxmh_profiles['Profile_Cluster'] == cluster_id]
    print(f"\n--- PROFILO CLUSTER {cluster_id} (n={len(cluster_data)} persone) ---")
    
    # Caratteristiche demografiche e di ascolto
    print("\nCaratteristiche Principali:")
    print(f"  - Età media: {cluster_data['Age'].mean():.1f} anni")
    print(f"  - Ore di ascolto medie: {cluster_data['Hours per day'].mean():.2f} ore/giorno")
    print(f"  - BPM medio: {cluster_data['BPM'].mean():.2f}")
    
    # Attività musicali
    instrumentalist_pct = cluster_data['Instrumentalist_numeric'].mean() * 100
    composer_pct = cluster_data['Composer_numeric'].mean() * 100
    print(f"  - Strumentisti: {instrumentalist_pct:.1f}%")
    print(f"  - Compositori: {composer_pct:.1f}%")
    
    # Generi più ascoltati
    genre_means = cluster_data[genre_numeric_cols].mean().sort_values(ascending=False)
    print("\nGeneri più caratteristici (punteggio medio di frequenza):")
    for i in range(5):
        genre_name = genre_means.index[i].replace('Frequency [', '').replace(']_numeric', '')
        print(f"  - {i+1}. {genre_name}: {genre_means.iloc[i]:.2f}")
        
    # ANALISI SALUTE MENTALE (l'obiettivo finale)
    mental_means = cluster_data[['Depression', 'Anxiety', 'Insomnia', 'OCD']].mean()
    print("\nProfilo di Salute Mentale Associato:")
    for condition, mean_val in mental_means.items():
        print(f"  - {condition:10}: {mean_val:.2f}")

# 5. CONFRONTO GRAFICO DELLA SALUTE MENTALE TRA CLUSTER
print("\n" + "="*80)
print("5. CONFRONTO GRAFICO DELLA SALUTE MENTALE TRA CLUSTER")
print("="*80)

cluster_mental_means = mxmh_profiles.groupby('Profile_Cluster')[['Depression', 'Anxiety', 'Insomnia', 'OCD']].mean()

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
fig.suptitle('Confronto Score di Salute Mentale per Profilo di Ascoltatore', fontsize=18)
mental_conditions = ['Depression', 'Anxiety', 'Insomnia', 'OCD']

for i, condition in enumerate(mental_conditions):
    ax = axes[i // 2, i % 2]
    means = cluster_mental_means[condition]
    bars = ax.bar(means.index, means.values, color=plt.cm.viridis(np.linspace(0, 1, best_k_profile)), alpha=0.8)
    
    ax.set_title(condition)
    ax.set_xlabel('Cluster Profilo')
    ax.set_ylabel('Score Medio')
    ax.set_xticks(means.index)
    ax.set_xticklabels([f'Profilo {i}' for i in means.index])
    
    # Aggiunta valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nAnalisi profili completata!")

# %%
# 6. VISUALIZZAZIONI AVANZATE DEI PROFILI CLUSTER
print("\n" + "="*80)
print("6. VISUALIZZAZIONI AVANZATE DEI PROFILI CLUSTER")
print("="*80)

# A. GRAFICO RADAR PER CONFRONTARE I PROFILI
print("\nGenerazione Grafico Radar...")

# Calcoliamo le medie delle feature per ogni cluster (usando i dati standardizzati)
profile_features_scaled_df = pd.DataFrame(X_profile, columns=profile_features_cols, index=profile_features.index)
profile_features_scaled_df['Profile_Cluster'] = profile_labels
cluster_means_scaled = profile_features_scaled_df.groupby('Profile_Cluster').mean()

# Selezioniamo un sottoinsieme di feature più interpretabili per il radar
radar_features = ['Age', 'Hours per day', 'BPM', 'Instrumentalist_numeric', 'Composer_numeric']
# Aggiungiamo i 3 generi con la varianza più alta per vedere le differenze
top_variance_genres = profile_features[genre_numeric_cols].var().nlargest(3).index.tolist()
radar_features.extend(top_variance_genres)

# Rimuoviamo '_numeric' e altri suffissi per etichette più pulite
labels_clean = [l.replace('_numeric', '').replace('Frequency [', '').replace(']', '') for l in radar_features]
angles = np.linspace(0, 2 * np.pi, len(labels_clean), endpoint=False).tolist()
angles += angles[:1] # Chiude il cerchio

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, row in cluster_means_scaled[radar_features].iterrows():
    data = row.tolist()
    data += data[:1]
    ax.plot(angles, data, label=f'Profilo {i}')
    ax.fill(angles, data, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_clean)
ax.set_title('Confronto Profili Cluster (Valori Standardizzati)', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()


# B. GRAFICI A BARRE PER PREFERENZE DI GENERE
print("\nGenerazione Grafici a Barre per Genere...")

# Numero di top generi da mostrare
top_n_genres = 5 

# Calcoliamo le medie non standardizzate per l'interpretazione
cluster_genre_means = mxmh_profiles.groupby('Profile_Cluster')[genre_numeric_cols].mean()

num_clusters = len(cluster_genre_means)
fig, axes = plt.subplots(1, num_clusters, figsize=(7 * num_clusters, 5), sharey=True)
if num_clusters == 1: axes = [axes]

for i, ax in enumerate(axes):
    cluster_data = cluster_genre_means.loc[i].sort_values(ascending=False).head(top_n_genres)
    clean_labels = [l.replace('Frequency [', '').replace(']_numeric', '') for l in cluster_data.index]
    
    bars = ax.barh(clean_labels, cluster_data.values, color=plt.cm.viridis(np.linspace(0.9, 0.4, top_n_genres)))
    ax.set_title(f'Profilo {i}\nTop {top_n_genres} Generi Preferiti')
    ax.set_xlabel('Frequenza Media di Ascolto (0-3)')
    ax.invert_yaxis() # Mostra il genere top in alto
    
    # Aggiungi valori
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

plt.suptitle('Preferenze Musicali per Profilo di Ascoltatore', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.show()


# C. BOX PLOT PER FEATURE NUMERICHE CHIAVE
print("\nGenerazione Box Plot per Feature Numeriche...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Distribuzione Feature Numeriche per Profilo', fontsize=16)

key_numeric_features = ['Age', 'Hours per day', 'BPM']

for i, feature in enumerate(key_numeric_features):
    sns.boxplot(data=mxmh_profiles, x='Profile_Cluster', y=feature, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribuzione di "{feature}"')
    axes[i].set_xlabel('Cluster Profilo')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
