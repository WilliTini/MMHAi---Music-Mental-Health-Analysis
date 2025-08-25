"""
APPROCCIO SEMPLICE: Mental Health Score Prediction
Predizione realistica con features essenziali e valutazione pratica
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Carica e prepara i dati con approccio semplice"""
    print("CARICAMENTO DATI")
    print("=" * 50)
    
    df = pd.read_csv('data/mxmh_final.csv')
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # Crea Mental Health Score (SOMMA è più intuitiva)
    mental_vars = ['Depression', 'Anxiety', 'Insomnia', 'OCD']
    df['Mental_Health_Score'] = df[mental_vars].sum(axis=1)
    
    print(f"\nMENTAL HEALTH SCORE:")
    print(f"   Range: {df['Mental_Health_Score'].min()} - {df['Mental_Health_Score'].max()}")
    print(f"   Media: {df['Mental_Health_Score'].mean():.1f}")
    print(f"   Std: {df['Mental_Health_Score'].std():.1f}")
    
    return df

def select_simple_features(df):
    """Seleziona solo le features più importanti e comprensibili"""
    print(f"\nSELEZIONE FEATURES SEMPLICI")
    print("=" * 40)
    
    # Features demografiche/comportamentali essenziali
    basic_features = [
        'Age', 'Hours per day',  # Demografi + comportamento
    ]
    
    # Top generi musicali (i più popolari)
    music_features = [
        'Classical', 'Pop', 'Rock', 'Hip hop', 'Jazz'  # I 5 generi principali
    ]
    
    # Features disponibili
    available_basic = [f for f in basic_features if f in df.columns]
    available_music = [f for f in music_features if f in df.columns]
    
    selected_features = available_basic + available_music
    
    print(f"Features demografiche: {available_basic}")
    print(f"Features musicali: {available_music}")
    print(f"TOTALE FEATURES: {len(selected_features)}")
    
    return selected_features

def evaluate_with_tolerance(y_true, y_pred, tolerance_pct):
    """Valuta le predizioni con tolerance range"""
    tolerance_range = np.abs(y_true * tolerance_pct / 100)
    within_tolerance = np.abs(y_true - y_pred) <= tolerance_range
    tolerance_accuracy = np.mean(within_tolerance)
    
    # MAE in unità comprensibili
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'tolerance_accuracy': tolerance_accuracy,
        'mae': mae,
        'r2': r2,
        'mean_error_points': mae  # Errore medio in "punti" di mental health
    }

def simple_prediction_analysis():
    """Analisi semplice e diretta"""
    print("\nMENTAL HEALTH PREDICTION - APPROCCIO SEMPLICE")
    print("=" * 60)
    
    # 1. Carica dati
    df = load_and_prepare_data()
    
    # 2. Seleziona features semplici
    features = select_simple_features(df)
    
    # 3. Prepara dati
    X = df[features].fillna(0)  # Riempi NaN con 0
    y = df['Mental_Health_Score']
    
    print(f"\nPREPARAZIONE DATI:")
    print(f"   Features shape: {X.shape}")
    print(f"   Target range: {y.min()} - {y.max()}")
    print(f"   Missing values: {X.isnull().sum().sum()}")
    
    # 4. Split e scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTRAIN/TEST SPLIT:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # 5. Modelli semplici
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    print(f"\nTRAINING MODELLI:")
    print("=" * 30)
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        if name == 'Ridge':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evalua con diverse tolerance
        tolerances = [10, 15, 20]
        model_results = {}
        
        for tol in tolerances:
            metrics = evaluate_with_tolerance(y_test, y_pred, tol)
            model_results[f'tolerance_{tol}'] = metrics
        
        results[name] = model_results
        
        # Print risultati del modello
        print(f"\nRISULTATI {name}:")
        for tol in tolerances:
            metrics = model_results[f'tolerance_{tol}']
            print(f"   Tolerance ±{tol}%:")
            print(f"     Accuracy: {metrics['tolerance_accuracy']:.1%}")
            print(f"     MAE: {metrics['mae']:.1f} punti")
            print(f"     R²: {metrics['r2']:.3f}")
    
    # 6. Confronto finale
    print(f"\nCONFRONTO FINALE:")
    print("=" * 40)
    
    for tolerance in [10, 15, 20]:
        print(f"\nTOLERANCE ±{tolerance}%:")
        for model_name in results:
            metrics = results[model_name][f'tolerance_{tolerance}']
            print(f"   {model_name:15}: {metrics['tolerance_accuracy']:.1%} accuracy, "
                  f"MAE {metrics['mae']:.1f} punti")
    
    # 7. Analisi pratica
    print(f"\nINTERPRETAZIONE PRATICA:")
    print("=" * 35)
    print("Se accuracy >= 30% con tolerance ±15%: BUONO")
    print("Se accuracy 20-30%: ACCETTABILE") 
    print("Se accuracy < 20%: INSUFFICIENTE")
    print()
    print("Mental Health Score range 0-40 punti")
    print("MAE < 5 punti = Errore ragionevole")
    print("MAE 5-8 punti = Errore accettabile")
    print("MAE > 8 punti = Errore eccessivo")
    
    return results

if __name__ == "__main__":
    results = simple_prediction_analysis()
    print("\nANALISI COMPLETATA!")
