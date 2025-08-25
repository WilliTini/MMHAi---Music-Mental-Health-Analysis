# PREPROCESSING SEMPLICE CON ANALISI BPM

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

print("PREPROCESSING MUSICA E SALUTE MENTALE")

# Carica dataset
mxmh = pd.read_csv("data/mxmh_survey_results.csv")
print(f"Dataset originale: {mxmh.shape}")

# Rimuovi colonne inutili
mxmh_clean = mxmh.drop(columns=['Permissions', 'Timestamp'], errors='ignore')
print(f"Rimosso Permissions e Timestamp: {mxmh_clean.shape}")

# Controllo e pulizia BPM
print(f"BPM range originale: {mxmh_clean['BPM'].min()} - {mxmh_clean['BPM'].max()}")
print(f"BPM valori anomali: {(mxmh_clean['BPM'] > 220).sum()}")

# Filtra BPM ragionevoli (50-220 BPM tipici per musica)
mxmh_clean = mxmh_clean[(mxmh_clean['BPM'] >= 50) & (mxmh_clean['BPM'] <= 220) | mxmh_clean['BPM'].isna()].copy()
print(f"Dataset dopo filtro BPM: {mxmh_clean.shape}")
print(f"BPM range filtrato: {mxmh_clean['BPM'].min()} - {mxmh_clean['BPM'].max()}")

# Converti generi musicali
map_freq = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Very frequently": 3}
genre_cols = [col for col in mxmh_clean.columns if col.startswith("Frequency [")]

print(f"Conversione {len(genre_cols)} generi musicali...")
for col in genre_cols:  
    mxmh_clean[col] = mxmh_clean[col].map(map_freq)

# Converti variabili binarie
binary_cols = ['Instrumentalist', 'Composer', 'While working', 'Exploratory', 'Foreign languages']
for col in binary_cols:
    if col in mxmh_clean.columns:
        mxmh_clean[col] = (mxmh_clean[col] == 'Yes').astype(int)

print("Conversioni completate")

# CLUSTERING SOLO SU SALUTE MENTALE 
mental_health_cols = ["Depression", "Anxiety", "Insomnia", "OCD"]

# Prendi dati senza valori mancanti per clustering
clustering_data = mxmh_clean[mental_health_cols].dropna()
print(f"Campione clustering: {len(clustering_data)} persone")

# Standardizzazione SOLO salute mentale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)

# Trova numero ottimale cluster
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette={score:.3f}")

best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Cluster ottimale: {best_k}")

# Clustering finale
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Aggiungi cluster al dataframe completo 
mxmh_clustered = mxmh_clean.loc[clustering_data.index].copy()
mxmh_clustered["Cluster"] = cluster_labels

print(f"\nDistribuzione cluster:")
for i in range(best_k):
    count = (cluster_labels == i).sum()
    print(f"Cluster {i}: {count} persone")

# Analisi cluster con BPM come variabile dipendente
print(f"\nAnalisi cluster (BPM come conseguenza):")
for cluster_id in range(best_k):
    cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
    
    print(f"\nCLUSTER {cluster_id}")
    print("-" * 30)
    
    # Salute mentale (variabili clustering)
    for col in mental_health_cols:
        mean_val = cluster_data[col].mean()
        print(f"{col}: {mean_val:.2f}")
    
    # BPM come variabile dipendente (NON usata per clustering)
    cluster_bpm = cluster_data['BPM'].dropna()
    if len(cluster_bpm) > 0:
        bpm_mean = cluster_bpm.mean()
        bpm_std = cluster_bpm.std()
        print(f"BPM: {bpm_mean:.1f} ± {bpm_std:.1f} (n={len(cluster_bpm)})")
    else:
        print("BPM: nessun dato")
    
    # ALTRE VARIABILI DIPENDENTI (conseguenze dei pattern di salute mentale)
    
    # Comportamenti musicali
    hours_mean = cluster_data['Hours per day'].mean()
    print(f"Ore ascolto: {hours_mean:.2f}")
    
    # Demografia
    age_mean = cluster_data['Age'].mean()
    print(f"Età: {age_mean:.1f}")
    
    # Comportamenti musicali binari (% del cluster)
    if 'While working' in cluster_data.columns:
        work_pct = cluster_data['While working'].mean() * 100
        print(f"Ascolta lavorando: {work_pct:.1f}%")
    
    if 'Instrumentalist' in cluster_data.columns:
        instr_pct = cluster_data['Instrumentalist'].mean() * 100
        print(f"Suona strumenti: {instr_pct:.1f}%")
    
    if 'Composer' in cluster_data.columns:
        comp_pct = cluster_data['Composer'].mean() * 100
        print(f"Compone musica: {comp_pct:.1f}%")
    
    if 'Exploratory' in cluster_data.columns:
        expl_pct = cluster_data['Exploratory'].mean() * 100
        print(f"Esplora nuova musica: {expl_pct:.1f}%")
    
    # Generi musicali preferiti (top 3)
    genre_means = {}
    for col in genre_cols:
        if col in cluster_data.columns:
            genre_name = col.replace('Frequency [', '').replace(']', '')
            genre_means[genre_name] = cluster_data[col].mean()
    
    if genre_means:
        top_genres = sorted(genre_means.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 generi: {', '.join([f'{g}({v:.1f})' for g, v in top_genres])}")
    
    # Effetti percepiti della musica
    if 'Music effects' in cluster_data.columns:
        effects = cluster_data['Music effects'].value_counts()
        if len(effects) > 0:
            top_effect = effects.index[0]
            effect_pct = (effects.iloc[0] / len(cluster_data)) * 100
            print(f"Effetto principale: {top_effect} ({effect_pct:.1f}%)")

# GRAFICI per analizzare tutte le variabili dipendenti
print(f"\nCreazione grafici per tutte le variabili dipendenti...")

plt.figure(figsize=(20, 15))

# 1. Profili salute mentale per cluster (VARIABILI INDIPENDENTI)
plt.subplot(3, 4, 1)
cluster_mental = mxmh_clustered.groupby("Cluster")[mental_health_cols].mean()
x_pos = np.arange(len(cluster_mental))
width = 0.2

colors = ['red', 'blue', 'green', 'orange']
for i, condition in enumerate(mental_health_cols):
    plt.bar(x_pos + i*width, cluster_mental[condition], width, 
            label=condition, alpha=0.7, color=colors[i])

plt.title('Profili Salute Mentale per Cluster\n(Variabili Indipendenti)')
plt.xlabel('Cluster')
plt.ylabel('Score Medio')
plt.xticks(x_pos + width*1.5, [f'C{i}' for i in cluster_mental.index])
plt.legend()
plt.grid(True, alpha=0.3)

# 2. BPM per cluster (boxplot) - VARIABILE DIPENDENTE
plt.subplot(3, 4, 2)
bpm_data = []
cluster_labels_for_bpm = []
for cluster_id in range(best_k):
    cluster_bpm = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]['BPM'].dropna()
    bpm_data.extend(cluster_bpm.tolist())
    cluster_labels_for_bpm.extend([f'C{cluster_id}'] * len(cluster_bpm))

if bpm_data:
    bpm_df = pd.DataFrame({'BPM': bpm_data, 'Cluster': cluster_labels_for_bpm})
    sns.boxplot(data=bpm_df, x='Cluster', y='BPM')
    plt.title('Distribuzione BPM per Cluster\n(Variabile Dipendente)')
    plt.ylabel('BPM')
else:
    plt.text(0.5, 0.5, 'Nessun dato BPM', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('BPM per Cluster - Nessun dato')

# 3. BPM medio per cluster (barplot) - VARIABILE DIPENDENTE
plt.subplot(3, 4, 3)
cluster_bpm_means = []
cluster_ids = []
for cluster_id in range(best_k):
    cluster_bpm = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]['BPM'].dropna()
    if len(cluster_bpm) > 0:
        cluster_bpm_means.append(cluster_bpm.mean())
        cluster_ids.append(cluster_id)

if cluster_bpm_means:
    plt.bar(range(len(cluster_bpm_means)), cluster_bpm_means, alpha=0.7, color='purple')
    plt.title('BPM Medio per Cluster\n(Variabile Dipendente)')
    plt.xlabel('Cluster')
    plt.ylabel('BPM Medio')
    plt.xticks(range(len(cluster_bpm_means)), [f'C{cid}' for cid in cluster_ids])
    
    # Aggiungi valori sulle barre
    for i, value in enumerate(cluster_bpm_means):
        plt.text(i, value + 2, f'{value:.1f}', ha='center', va='bottom')
else:
    plt.text(0.5, 0.5, 'Nessun dato BPM', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('BPM Medio - Nessun dato')

# 4. Ore ascolto per cluster
plt.subplot(3, 4, 4)
cluster_hours = mxmh_clustered.groupby("Cluster")['Hours per day'].mean()
plt.bar(range(len(cluster_hours)), cluster_hours.values, color='darkblue', alpha=0.7)
plt.title('Ore Ascolto per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Ore/Giorno')
plt.xticks(range(len(cluster_hours)), [f'C{i}' for i in cluster_hours.index])

# 5. Scatter: Score Salute Mentale vs BPM
plt.subplot(3, 4, 5)
for cluster_id in range(best_k):
    cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
    mental_score = cluster_data[mental_health_cols].mean(axis=1)
    cluster_bpm = cluster_data['BPM'].dropna()
    
    # Solo se ci sono dati BPM per questo cluster
    if len(cluster_bpm) > 0:
        # Prendi solo i mental_score corrispondenti ai BPM disponibili
        mental_score_with_bpm = mental_score[cluster_data['BPM'].notna()]
        plt.scatter(mental_score_with_bpm, cluster_bpm, 
                   label=f'Cluster {cluster_id}', alpha=0.6, s=30)

plt.xlabel('Score Salute Mentale')
plt.ylabel('BPM')
plt.title('Relazione Salute Mentale - BPM')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Età per cluster
plt.subplot(3, 4, 6)
cluster_age = mxmh_clustered.groupby("Cluster")['Age'].mean()
plt.bar(range(len(cluster_age)), cluster_age.values, color='orange', alpha=0.7)
plt.title('Età Media per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Età')
plt.xticks(range(len(cluster_age)), [f'C{i}' for i in cluster_age.index])

# 7. Comportamenti musicali (% per cluster)
plt.subplot(3, 4, 7)
behaviors = ['While working', 'Instrumentalist', 'Composer', 'Exploratory']
behavior_data = []
cluster_names = []

for cluster_id in range(best_k):
    cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
    cluster_behaviors = []
    for behavior in behaviors:
        if behavior in cluster_data.columns:
            pct = cluster_data[behavior].mean() * 100
            cluster_behaviors.append(pct)
        else:
            cluster_behaviors.append(0)
    behavior_data.append(cluster_behaviors)
    cluster_names.append(f'C{cluster_id}')

x = np.arange(len(behaviors))
width = 0.35
colors = ['skyblue', 'lightcoral']

for i, (cluster_behavior, color) in enumerate(zip(behavior_data, colors)):
    plt.bar(x + i*width, cluster_behavior, width, label=cluster_names[i], 
            color=color, alpha=0.7)

plt.title('Comportamenti Musicali per Cluster (%)')
plt.xlabel('Comportamento')
plt.ylabel('Percentuale')
plt.xticks(x + width/2, [b.replace(' ', '\n') for b in behaviors], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Top 3 generi per cluster
plt.subplot(3, 4, 8)
top_genres_by_cluster = {}
for cluster_id in range(best_k):
    cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
    genre_means = {}
    for col in genre_cols:
        if col in cluster_data.columns:
            genre_name = col.replace('Frequency [', '').replace(']', '')
            genre_means[genre_name] = cluster_data[col].mean()
    
    if genre_means:
        top_3 = sorted(genre_means.items(), key=lambda x: x[1], reverse=True)[:3]
        top_genres_by_cluster[cluster_id] = top_3

# Visualizza top generi
if top_genres_by_cluster:
    cluster_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    y_pos = 0
    for cluster_id, top_genres in top_genres_by_cluster.items():
        for i, (genre, score) in enumerate(top_genres):
            plt.barh(y_pos, score, color=cluster_colors[cluster_id % len(cluster_colors)], 
                    alpha=0.7, label=f'C{cluster_id}' if i == 0 else "")
            plt.text(score + 0.05, y_pos, f'{genre[:8]}', va='center', fontsize=8)
            y_pos += 1
        y_pos += 0.5  # Spazio tra cluster

plt.title('Top 3 Generi per Cluster')
plt.xlabel('Score Frequenza')
plt.ylabel('Ranking')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Distribuzione generale BPM
plt.subplot(3, 4, 9)
all_bpm = mxmh_clustered['BPM'].dropna()
if len(all_bpm) > 0:
    plt.hist(all_bpm, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(all_bpm.mean(), color='red', linestyle='--', 
               label=f'Media: {all_bpm.mean():.1f}')
    plt.xlabel('BPM')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione BPM nel Dataset')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Nessun dato BPM', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Distribuzione BPM - Nessun dato')

# 10. Effetti della musica per cluster
plt.subplot(3, 4, 10)
if 'Music effects' in mxmh_clustered.columns:
    effect_counts = {}
    for cluster_id in range(best_k):
        cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
        effects = cluster_data['Music effects'].value_counts(normalize=True) * 100
        effect_counts[cluster_id] = effects
    
    # Prendi i 3 effetti più comuni
    all_effects = mxmh_clustered['Music effects'].value_counts()
    top_effects = all_effects.index[:3]
    
    x = np.arange(len(top_effects))
    width = 0.35
    colors = ['skyblue', 'lightcoral']
    
    for i, cluster_id in enumerate(range(best_k)):
        cluster_percentages = []
        for effect in top_effects:
            pct = effect_counts[cluster_id].get(effect, 0)
            cluster_percentages.append(pct)
        
        plt.bar(x + i*width, cluster_percentages, width, 
               label=f'Cluster {cluster_id}', color=colors[i], alpha=0.7)
    
    plt.title('Effetti Musica per Cluster (%)')
    plt.xlabel('Effetto')
    plt.ylabel('Percentuale')
    plt.xticks(x + width/2, [e[:8] for e in top_effects], rotation=45)
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Nessun dato effetti', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Effetti Musica - Nessun dato')

# 11. Servizi streaming per cluster
plt.subplot(3, 4, 11)
if 'Primary streaming service' in mxmh_clustered.columns:
    streaming_counts = {}
    for cluster_id in range(best_k):
        cluster_data = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]
        streaming = cluster_data['Primary streaming service'].value_counts(normalize=True) * 100
        streaming_counts[cluster_id] = streaming
    
    # Prendi i 3 servizi più comuni
    all_streaming = mxmh_clustered['Primary streaming service'].value_counts()
    top_streaming = all_streaming.index[:3]
    
    x = np.arange(len(top_streaming))
    width = 0.35
    colors = ['skyblue', 'lightcoral']
    
    for i, cluster_id in enumerate(range(best_k)):
        cluster_percentages = []
        for service in top_streaming:
            pct = streaming_counts[cluster_id].get(service, 0)
            cluster_percentages.append(pct)
        
        plt.bar(x + i*width, cluster_percentages, width, 
               label=f'Cluster {cluster_id}', color=colors[i], alpha=0.7)
    
    plt.title('Servizi Streaming per Cluster (%)')
    plt.xlabel('Servizio')
    plt.ylabel('Percentuale')
    plt.xticks(x + width/2, [s[:8] for s in top_streaming], rotation=45)
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Nessun dato streaming', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Servizi Streaming - Nessun dato')

# 12. Correlazione tra tutte le variabili dipendenti
plt.subplot(3, 4, 12)
dependent_vars = ['BPM', 'Hours per day', 'Age']
behavioral_vars = ['While working', 'Instrumentalist', 'Composer', 'Exploratory']

# Aggiungi variabili comportamentali se esistono
correlation_vars = dependent_vars.copy()
for var in behavioral_vars:
    if var in mxmh_clustered.columns:
        correlation_vars.append(var)

if len(correlation_vars) > 2:
    corr_data = mxmh_clustered[correlation_vars].corr()
    im = plt.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    plt.title('Correlazioni Variabili Dipendenti')
    plt.xticks(range(len(correlation_vars)), 
              [v.replace(' ', '\n') for v in correlation_vars], rotation=45, fontsize=8)
    plt.yticks(range(len(correlation_vars)), 
              [v.replace(' ', '\n') for v in correlation_vars], fontsize=8)
    
    # Aggiungi valori di correlazione
    for i in range(len(correlation_vars)):
        for j in range(len(correlation_vars)):
            plt.text(j, i, f'{corr_data.iloc[i,j]:.2f}', 
                    ha='center', va='center', fontsize=8)
else:
    plt.text(0.5, 0.5, 'Dati insufficienti\nper correlazioni', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Correlazioni - Dati insufficienti')

plt.tight_layout()
plt.show()

# Analisi statistica BPM vs Cluster
print(f"\nAnalisi statistica BPM:")
bpm_by_cluster = {}
for cluster_id in range(best_k):
    cluster_bpm = mxmh_clustered[mxmh_clustered["Cluster"] == cluster_id]['BPM'].dropna()
    if len(cluster_bpm) > 0:
        bpm_by_cluster[cluster_id] = cluster_bpm.tolist()
        print(f"Cluster {cluster_id} BPM: n={len(cluster_bpm)}, media={cluster_bpm.mean():.1f}, std={cluster_bpm.std():.1f}")

# Correlazioni cluster-BPM
if len(bpm_by_cluster) > 1:
    print(f"\nI cluster mostrano pattern BPM diversi:")
    cluster_means = [np.mean(bpm_by_cluster[cid]) for cid in sorted(bpm_by_cluster.keys())]
    print(f"Range BPM tra cluster: {min(cluster_means):.1f} - {max(cluster_means):.1f}")
    print(f"Differenza max: {max(cluster_means) - min(cluster_means):.1f} BPM")

# Salva risultati
mxmh_clustered.to_csv("data/mxmh_final.csv", index=False)
print(f"\nSalvato in data/mxmh_final.csv")
print("ANALISI COMPLETATA!")
