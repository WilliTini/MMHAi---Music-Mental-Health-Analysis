"""
Final Classifier Optimization - Music and Mental Health
Approccio mirato con feature engineering semplificato, GridSearchCV e LightGBM.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Importa XGBoost e LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost disponibile.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost non disponibile.")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("LightGBM disponibile.")
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM non disponibile.")

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
    
    # Pulisci nomi colonne per compatibilità
    clean_feature_names = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_') for col in X.columns]
    X.columns = clean_feature_names
    
    # Gestisci valori mancanti
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    print(f"Dataset preparato: {X.shape[0]} campioni, {X.shape[1]} features.")
    return X, y

def simplified_feature_engineering(X):
    """Feature engineering semplificato, senza feature polinomiali."""
    print("\n=== FEATURE ENGINEERING SEMPLIFICATO ===")
    X_enhanced = X.copy()
    
    # Aggiungi solo feature di rapporto e statistiche, che sono più robuste
    numeric_cols = X_enhanced.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:
                if (X_enhanced[col2] != 0).all():
                    X_enhanced[f'{col1}_{col2}_ratio'] = X_enhanced[col1] / (X_enhanced[col2] + 1e-8)
        
        X_enhanced['features_mean'] = X_enhanced[numeric_cols].mean(axis=1)
        X_enhanced['features_std'] = X_enhanced[numeric_cols].std(axis=1)
    
    print(f"Feature engineering completato: {X_enhanced.shape[1]} features finali.")
    return X_enhanced

def advanced_feature_selection(X, y):
    """Selezione feature robusta."""
    print("\n=== SELEZIONE FEATURES AVANZATA ===")
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # K-Best
    selector_k = SelectKBest(f_classif, k=min(15, X.shape[1]))
    selector_k.fit(X, y)
    selected_k = X.columns[selector_k.get_support()].tolist()
    
    # RFE
    rfe = RFE(estimator, n_features_to_select=min(10, X.shape[1]))
    rfe.fit(X, y)
    selected_rfe = X.columns[rfe.get_support()].tolist()
    
    # Model-based
    selector_model = SelectFromModel(estimator, threshold='median')
    selector_model.fit(X, y)
    selected_model = X.columns[selector_model.get_support()].tolist()
    
    final_features = list(set(selected_k + selected_rfe + selected_model))
    print(f"Features finali combinate: {len(final_features)}")
    return X[final_features], final_features

def final_classifier_optimization():
    """Esegue l'ottimizzazione finale del classificatore."""
    print("Ottimizzazione finale classificatore")
    print("="*80)
    
    X, y = load_and_prepare_data()
    X_engineered = simplified_feature_engineering(X)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_engineered), columns=X_engineered.columns, index=X_engineered.index)
    
    X_selected, selected_features = advanced_feature_selection(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Modelli e griglie di parametri per GridSearchCV
    models_and_params = {}
    
    # RandomForest
    models_and_params['RandomForest'] = (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    )
    
    # GradientBoosting
    models_and_params['GradientBoosting'] = (
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    )
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models_and_params['XGBoost'] = (
            xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        )
        
    # LightGBM
    if LGBM_AVAILABLE:
        models_and_params['LightGBM'] = (
            lgb.LGBMClassifier(random_state=42),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [20, 31]
            }
        )

    # Esegui GridSearchCV per ogni modello
    best_estimators = {}
    for name, (model, params) in models_and_params.items():
        print(f"\nOttimizzando {name} con GridSearchCV...")
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimators[name] = grid_search.best_estimator_
        print(f"  Miglior score CV per {name}: {grid_search.best_score_:.4f}")

    # Valuta i modelli migliori sul test set
    results = {}
    for name, model in best_estimators.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'model': model}
        print(f"{name} Test Accuracy: {accuracy:.4f}")

    # Ensemble
    if len(best_estimators) >= 2:
        print("\nCreando ensemble...")
        ensemble = VotingClassifier(estimators=list(best_estimators.items()), voting='soft')
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        results['ENSEMBLE'] = {'accuracy': accuracy_ensemble, 'model': ensemble}
        print(f"Ensemble Test Accuracy: {accuracy_ensemble:.4f}")

    # Report finale
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "="*80)
    print("Risultati finali ottimizzazione")
    print("="*80)
    print(f"Miglior modello: {best_model_name}")
    print(f"Miglior accuracy: {best_accuracy:.4f}")
    print(f"Features usate: {len(selected_features)}")

if __name__ == "__main__":
    final_classifier_optimization()
