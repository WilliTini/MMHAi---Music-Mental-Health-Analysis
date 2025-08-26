"""
Simple Classifier - Music and Mental Health
Approccio di classificazione di base senza feature engineering per stabilire una baseline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Carica e prepara i dati per la classificazione."""
    print("Caricamento dati...")
    df = pd.read_csv('data/mxmh_final.csv')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Rimuovi target e variabili correlate per evitare data leakage
    targets_and_leaks = ['Cluster', 'Depression', 'Anxiety', 'Insomnia', 'OCD', 'Music_effects_scaled', 'Mental_Health_Score']
    feature_cols = [col for col in numeric_cols if col not in targets_and_leaks]
    
    X = df[feature_cols].copy()
    y = df['Cluster'].copy()
    
    # Pulisci nomi colonne per compatibilità con alcuni modelli
    clean_feature_names = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_') for col in X.columns]
    X.columns = clean_feature_names
    
    # Gestisci valori mancanti
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    print(f"Dataset preparato: {X.shape[0]} campioni, {X.shape[1]} features.")
    print("Features utilizzate:", list(X.columns))
    return X, y

def simple_classification_pipeline():
    """Esegue una pipeline di classificazione semplice senza feature engineering."""
    print("Esecuzione classificatore semplice (senza feature engineering)")
    print("="*80)
    
    # 1. Carica i dati
    X, y = load_and_prepare_data()
    
    # 2. Suddividi i dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scala le feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=== ADDESTRAMENTO E VALUTAZIONE MODELLI ===")
    
    # Modelli da testare
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Addestra e valuta ogni modello
    for name, model in models.items():
        print(f"\nAddestrando {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  -> Accuracy di {name} sul test set: {accuracy:.4f}")
        
        # Valutazione con cross-validation per robustezza
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"  -> Accuracy media in Cross-Validation (5-fold): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    print("\n" + "="*80)
    print("Esecuzione completata.")
    print("Questo script fornisce una baseline di performance usando solo le feature originali.")

if __name__ == "__main__":
    simple_classification_pipeline()
