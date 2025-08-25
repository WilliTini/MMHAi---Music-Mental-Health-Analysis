"""
Musical Diversity Score Analysis - VERSIONE FINALE CORRETTA
Calcola quanto una persona esplora generi al di fuori delle sue preferenze principali
CORREGGE: Esclude sempre il Fav genre dichiarato dalla persona (usando mxmh_final.csv con valori numerici)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def calculate_diversity_scores_final(df, top_n_genres=3):
    """
    VERSIONE FINALE CORRETTA - Calcola diversity score escludendo il Fav genre dichiarato
    
    Args:
        df: DataFrame con dati musicali (mxmh_final.csv con valori già numerici)
        top_n_genres: Numero totale di generi da escludere (1=solo fav, 2=fav+1, 3=fav+2)
    
    Returns:
        DataFrame con diversity scores corretti
    """
    print(f"CALCOLO DIVERSITY SCORE FINALE (ESCLUDE FAV + TOP {top_n_genres-1})")
    print("=" * 60)
    
    # Identifica colonne dei generi
    genre_cols = [col for col in df.columns if col.startswith('Frequency [') and col.endswith(']')]
    print(f"Generi trovati: {len(genre_cols)}")
    
    # Mapping da nomi generi a colonne frequency
    genre_mapping = {}
    for col in genre_cols:
        genre_name = col.replace('Frequency [', '').replace(']', '')
        genre_mapping[genre_name] = col
        
    # Alcuni nomi potrebbero essere diversi tra Fav genre e colonne frequency
    # Aggiungi mapping per casi particolari
    alternative_mappings = {
        'Hip hop': 'Hip hop',
        'K pop': 'K pop', 
        'R&B': 'R&B',
        'Video game music': 'Video game music'
    }
    
    print(f"Genre mapping: {list(genre_mapping.keys())}")
    
    # Inizializza risultati
    diversity_scores = []
    excluded_genres_info = []
    
    print("Calcolando diversity score per ogni persona...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:  # Progress indicator
            print(f"   Processando persona {idx}/{len(df)}...")
        
        # Estrai punteggi dei generi per questa persona (già numerici!)
        genre_scores = row[genre_cols].fillna(0)
        
        # 1. SEMPRE escludere il Fav genre dichiarato
        fav_genre = row['Fav genre']
        excluded_genres = []
        
        # Trova la colonna del genere favorito
        fav_genre_col = None
        
        # Prova match esatto
        if fav_genre in genre_mapping:
            fav_genre_col = genre_mapping[fav_genre]
        else:
            # Prova match case-insensitive
            for genre_name, col_name in genre_mapping.items():
                if genre_name.lower() == fav_genre.lower():
                    fav_genre_col = col_name
                    break
        
        if fav_genre_col is not None:
            excluded_genres.append(fav_genre_col)
        else:
            print(f"Attenzione: Genere favorito '{fav_genre}' non trovato per persona {idx}")
            # Se non trova il fav genre, salta questa persona o usa metodo alternativo
            # Per ora usiamo il genere con punteggio più alto come proxy
            fav_genre_col = genre_scores.idxmax()
            excluded_genres.append(fav_genre_col)
        
        # 2. Se richiesto, escludi anche altri generi con punteggi più alti
        if top_n_genres > 1:
            # Rimuovi il fav genre già escluso dai candidati
            remaining_scores = genre_scores.drop(excluded_genres)
            # Prendi i top (n-1) generi rimanenti
            additional_top = remaining_scores.nlargest(top_n_genres - 1).index
            excluded_genres.extend(list(additional_top))
        
        # 3. Calcola medie
        excluded_scores = genre_scores[excluded_genres]
        included_scores = genre_scores.drop(excluded_genres)
        
        excluded_mean = excluded_scores.mean() if len(excluded_scores) > 0 else 0
        included_mean = included_scores.mean() if len(included_scores) > 0 else 0
        
        # 4. Diversity Score: rapporto tra generi inclusi e esclusi
        if excluded_mean > 0:
            diversity_ratio = included_mean / excluded_mean
        else:
            diversity_ratio = 0  # Se non ascolta i generi esclusi
        
        # Alternative scores
        diversity_absolute = included_mean
        total_mean = genre_scores.mean()
        diversity_normalized = included_mean / total_mean if total_mean > 0 else 0
        
        diversity_scores.append({
            'diversity_ratio': diversity_ratio,
            'diversity_absolute': diversity_absolute,
            'diversity_normalized': diversity_normalized,
            'excluded_mean': excluded_mean,
            'included_mean': included_mean,
            'total_listening': genre_scores.sum(),
            'n_excluded_genres': len(excluded_genres)
        })
        
        # Salva info sui generi esclusi
        excluded_genre_names = [col.replace('Frequency [', '').replace(']', '') 
                               for col in excluded_genres]
        excluded_genres_info.append({
            'person_id': idx,
            'fav_genre': fav_genre,
            'excluded_genres': excluded_genre_names,
            'excluded_scores': [genre_scores[col] for col in excluded_genres]
        })
    
    # Converti in DataFrame
    diversity_df = pd.DataFrame(diversity_scores)
    
    # Statistiche
    print(f"\nDIVERSITY SCORE STATISTICS (FINALE):")
    print(f"  Ratio score - Mean: {diversity_df['diversity_ratio'].mean():.3f}, Std: {diversity_df['diversity_ratio'].std():.3f}")
    print(f"  Absolute score - Mean: {diversity_df['diversity_absolute'].mean():.3f}, Std: {diversity_df['diversity_absolute'].std():.3f}")
    print(f"  Normalized score - Mean: {diversity_df['diversity_normalized'].mean():.3f}, Std: {diversity_df['diversity_normalized'].std():.3f}")
    
    # Controllo qualità
    print(f"\nQUALITY CHECK:")
    print(f"  Persone con diversity_ratio = 0: {(diversity_df['diversity_ratio'] == 0).sum()}")
    print(f"  Media generi esclusi per persona: {diversity_df['n_excluded_genres'].mean():.1f}")
    print(f"  Range diversity_ratio: [{diversity_df['diversity_ratio'].min():.3f}, {diversity_df['diversity_ratio'].max():.3f}]")
    
    return diversity_df, excluded_genres_info

def analyze_diversity_by_cluster_final(df, diversity_df):
    """Analizza diversity score per cluster di salute mentale"""
    print(f"\nDIVERSITY ANALYSIS BY CLUSTER (FINALE)")
    print("=" * 50)
    
    # Combina dataframes
    combined = pd.concat([df[['Cluster']], diversity_df], axis=1)
    
    # Analisi per cluster
    for cluster in sorted(combined['Cluster'].unique()):
        cluster_data = combined[combined['Cluster'] == cluster]
        
        print(f"\nCluster {cluster}:")
        print(f"  N persone: {len(cluster_data)}")
        print(f"  Diversity ratio: {cluster_data['diversity_ratio'].mean():.3f} (±{cluster_data['diversity_ratio'].std():.3f})")
        print(f"  Diversity absolute: {cluster_data['diversity_absolute'].mean():.3f}")
        print(f"  Excluded genres mean: {cluster_data['excluded_mean'].mean():.3f}")
        print(f"  Included genres mean: {cluster_data['included_mean'].mean():.3f}")
    
    # Test statistico
    cluster_0 = combined[combined['Cluster'] == 0]['diversity_ratio']
    cluster_1 = combined[combined['Cluster'] == 1]['diversity_ratio']
    
    t_stat, p_value = ttest_ind(cluster_0, cluster_1)
    
    print(f"\nStatistical test (t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("  Significant difference in diversity between clusters!")
        diff_pct = ((cluster_0.mean() - cluster_1.mean()) / cluster_1.mean()) * 100
        print(f"  Cluster 0 ha {diff_pct:+.1f}% diversity rispetto a Cluster 1")
    else:
        print("  No significant difference between clusters")

def test_diversity_impact_final(df, diversity_df, diversity_type='diversity_ratio'):
    """Testa impatto diversity score sui modelli ML"""
    print(f"\nTESTING DIVERSITY IMPACT: {diversity_type} (FINALE)")
    print("=" * 50)
    
    # Prepara features base
    feature_cols = ['Age', 'Hours per day', 'While working', 'Instrumentalist', 
                   'Composer', 'Exploratory', 'Foreign languages']
    
    X_base = df[feature_cols].copy()
    X_with_diversity = pd.concat([X_base, diversity_df[[diversity_type]]], axis=1)
    
    # Rimuovi NaN values
    X_base = X_base.fillna(X_base.mean())
    X_with_diversity = X_with_diversity.fillna(X_with_diversity.mean())
    
    y_class = df['Cluster']  # Classification target
    y_reg = df['Anxiety'] + df['Depression'] + df['Insomnia'] + df['OCD']  # Regression target
    
    print(f"Comparison setup:")
    print(f"  Features WITHOUT diversity: {X_base.shape[1]}")
    print(f"  Features WITH diversity: {X_with_diversity.shape[1]}")
    
    # Split data
    X_base_train, X_base_test, y_class_train, y_class_test = train_test_split(
        X_base, y_class, test_size=0.2, random_state=42)
    X_div_train, X_div_test, _, _ = train_test_split(
        X_with_diversity, y_class, test_size=0.2, random_state=42)
    
    _, _, y_reg_train, y_reg_test = train_test_split(
        X_base, y_reg, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_base = StandardScaler()
    scaler_div = StandardScaler()
    
    X_base_train_scaled = scaler_base.fit_transform(X_base_train)
    X_base_test_scaled = scaler_base.transform(X_base_test)
    X_div_train_scaled = scaler_div.fit_transform(X_div_train)
    X_div_test_scaled = scaler_div.transform(X_div_test)
    
    # === CLASSIFICATION ===
    print(f"\nCLASSIFICATION COMPARISON:")
    
    # Without diversity
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores_base = cross_val_score(rf_base, X_base_train_scaled, y_class_train, cv=5)
    rf_base.fit(X_base_train_scaled, y_class_train)
    test_acc_base = accuracy_score(y_class_test, rf_base.predict(X_base_test_scaled))
    
    # With diversity
    rf_div = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores_div = cross_val_score(rf_div, X_div_train_scaled, y_class_train, cv=5)
    rf_div.fit(X_div_train_scaled, y_class_train)
    test_acc_div = accuracy_score(y_class_test, rf_div.predict(X_div_test_scaled))
    
    print(f"  WITHOUT Diversity:")
    print(f"    CV Accuracy: {cv_scores_base.mean():.3f} (±{cv_scores_base.std():.3f})")
    print(f"    Test Accuracy: {test_acc_base:.3f}")
    print(f"  WITH Diversity:")
    print(f"    CV Accuracy: {cv_scores_div.mean():.3f} (±{cv_scores_div.std():.3f})")
    print(f"    Test Accuracy: {test_acc_div:.3f}")
    
    acc_improvement = test_acc_div - test_acc_base
    acc_improvement_pct = (acc_improvement / test_acc_base) * 100
    print(f"  DIVERSITY IMPACT: {acc_improvement:+.3f} ({acc_improvement_pct:+.1f}%)")
    print(f"    {'Diversity score aiuta' if acc_improvement > 0 else 'Diversity score non aiuta'}")
    
    # === REGRESSION ===
    print(f"\nREGRESSION COMPARISON:")
    
    # Without diversity  
    lr_base = LinearRegression()
    cv_scores_base_reg = cross_val_score(lr_base, X_base_train_scaled, y_reg_train, cv=5, scoring='r2')
    lr_base.fit(X_base_train_scaled, y_reg_train)
    test_r2_base = r2_score(y_reg_test, lr_base.predict(X_base_test_scaled))
    
    # With diversity
    lr_div = LinearRegression()
    cv_scores_div_reg = cross_val_score(lr_div, X_div_train_scaled, y_reg_train, cv=5, scoring='r2')
    lr_div.fit(X_div_train_scaled, y_reg_train)
    test_r2_div = r2_score(y_reg_test, lr_div.predict(X_div_test_scaled))
    
    print(f"  WITHOUT Diversity:")
    print(f"    CV R²: {cv_scores_base_reg.mean():.3f} (±{cv_scores_base_reg.std():.3f})")
    print(f"    Test R²: {test_r2_base:.3f}")
    print(f"  WITH Diversity:")
    print(f"    CV R²: {cv_scores_div_reg.mean():.3f} (±{cv_scores_div_reg.std():.3f})")
    print(f"    Test R²: {test_r2_div:.3f}")
    
    r2_improvement = test_r2_div - test_r2_base
    if test_r2_base > 0:
        r2_improvement_pct = (r2_improvement / test_r2_base) * 100
    else:
        r2_improvement_pct = float('inf') if r2_improvement > 0 else 0
    
    print(f"  DIVERSITY IMPACT: {r2_improvement:+.3f} ({r2_improvement_pct:+.1f}%)")
    print(f"    {'Diversity score migliora significativamente!' if r2_improvement > 0.01 else 'Nessun miglioramento significativo'}")
    
    # Feature importance
    feature_names = feature_cols + [diversity_type]
    feature_importance = rf_div.feature_importances_
    
    return {
        'classification': {'base': test_acc_base, 'with_diversity': test_acc_div, 'improvement': acc_improvement},
        'regression': {'base': test_r2_base, 'with_diversity': test_r2_div, 'improvement': r2_improvement},
        'feature_importance': dict(zip(feature_names, feature_importance))
    }

def main():
    """Analisi principale diversity score FINALE CORRETTA"""
    print("MUSICAL DIVERSITY SCORE ANALYSIS - VERSIONE FINALE CORRETTA")
    print("=" * 80)
    print("Usa mxmh_final.csv (valori già numerici)")
    print("Esclude sempre il Fav genre dichiarato + eventualmente altri generi top")
    print("Gestisce match case-insensitive tra Fav genre e colonne frequency")
    print("=" * 80)
    
    # Carica dati
    df = pd.read_csv('data/mxmh_final.csv')
    print(f"Dataset loaded: {len(df)} people")
    
    # Controlla che abbiamo la colonna Fav genre
    if 'Fav genre' not in df.columns:
        print("ERRORE: Colonna 'Fav genre' non trovata!")
        return
    
    print(f"Generi favoriti più comuni:")
    print(df['Fav genre'].value_counts().head())
    
    # Test diversi valori di esclusioni
    results_summary = {}
    
    for exclude_n in [1, 2, 3]:
        print(f"\n{'='*30} EXCLUDE {exclude_n} GENRES {'='*30}")
        if exclude_n == 1:
            print("(Solo Fav genre dichiarato)")
        else:
            print(f"(Fav genre + {exclude_n-1} generi con punteggi più alti)")
        
        # Calcola diversity scores
        diversity_df, excluded_info = calculate_diversity_scores_final(df, top_n_genres=exclude_n)
        
        # Analizza per cluster
        analyze_diversity_by_cluster_final(df, diversity_df)
        
        # Test impatto su modelli (solo per exclude_3)
        if exclude_n == 3:
            print(f"\nTESTING ML IMPACT (EXCLUDE 3 only):")
            
            # Test diversi tipi di diversity score
            for div_type in ['diversity_ratio', 'diversity_absolute']:
                print(f"\n--- {div_type.upper()} ---")
                result = test_diversity_impact_final(df, diversity_df, div_type)
                results_summary[f'{exclude_n}_{div_type}'] = result
    
    # Summary finale
    print(f"\n" + "=" * 80)
    print("DIVERSITY SCORE SUMMARY - VERSIONE FINALE CORRETTA")
    print("=" * 80)
    print("Diversity score calcolato escludendo sempre il Fav genre dichiarato")
    print("Testato con esclusione di 1, 2, e 3 generi")
    print("Analizzate differenze tra cluster di salute mentale")
    print("Testato impatto su modelli di machine learning")
    
    print("\nMAIN FINDINGS:")
    print("   - Metodo corretto: esclude Fav genre dichiarato, non solo top per punteggio")
    print("   - Usa dataset mxmh_final.csv con valori già numerici")
    print("   - Gestisce match tra nomi di generi anche con case diverse")

if __name__ == "__main__":
    main()
