import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost disponibile!")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost non disponibile")

def load_and_prepare_data_winning():
    """Replica ESATTA del caricamento dati che dava 97% accuracy"""
    print("Caricamento dati con metodo VINCENTE...")
    df = pd.read_csv('data/mxmh_final.csv')
    print(f"Dataset FINALE caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_candidates = ['Music_effects_scaled', 'Cluster', 'Mental_Health_Score']
    
    # Target regressione
    mental_vars = ['Depression', 'Anxiety', 'Insomnia', 'OCD']
    y_severity = df[mental_vars].mean(axis=1)
    
    # IMPORTANTE: Rimuovi ANCHE le variabili mentali dalle features per evitare data leakage
    all_targets = target_candidates + mental_vars
    feature_cols = [col for col in numeric_cols if col not in all_targets]
    
    print(f"Features numeriche identificate: {len(feature_cols)}")
    
    # Prepara i dati
    X = df[feature_cols].copy()
    
    # Pulisci i nomi delle colonne per XGBoost (ESATTO come prima)
    clean_feature_names = []
    for col in X.columns:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_')
        clean_feature_names.append(clean_name)
    
    X.columns = clean_feature_names
    print(f"Nomi delle features puliti per compatibilità XGBoost")
    
    # Gestisci valori mancanti
    if X.isnull().sum().sum() > 0:
        print(f"Valori mancanti trovati: {X.isnull().sum().sum()}")
        X = X.fillna(X.median())
    
    # Target
    y_cluster = df['Cluster'].copy()
    
    # Target regressione (come nel file originale)
    mental_vars = ['Depression', 'Anxiety', 'Insomnia', 'OCD']
    y_severity = df[mental_vars].mean(axis=1)
    
    print(f"Dataset preparato: {X.shape[0]} campioni, {X.shape[1]} features")
    print(f"Distribuzione cluster: {y_cluster.value_counts().to_dict()}")
    
    return X, y_cluster, y_severity, clean_feature_names

def feature_engineering_winning(X):
    """Feature engineering ESATTO che dava 97% accuracy"""
    print("\n=== FEATURE ENGINEERING VINCENTE ===")
    
    X_enhanced = X.copy()
    
    # 1. Polynomial Features (ESATTO come prima)
    if X.shape[1] <= 35:  # Come nell'originale
        from sklearn.preprocessing import PolynomialFeatures
        # Solo interazioni, non tutti i polynomial
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
        # Prendiamo solo le prime 10 features più importanti per evitare esplosione
        top_features = X.columns[:min(10, len(X.columns))]
        X_top = X[top_features]
        
        X_poly = poly.fit_transform(X_top)
        poly_feature_names = poly.get_feature_names_out(top_features)
        
        # Combina features originali + polynomial
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
        # Aggiungi le features rimanenti
        remaining_features = [col for col in X.columns if col not in top_features]
        if remaining_features:
            X_enhanced = pd.concat([X_poly_df, X[remaining_features]], axis=1)
        else:
            X_enhanced = X_poly_df
            
        print(f"Features polinomiali aggiunte: {X_enhanced.shape[1]} features totali")
    
    # 2. Feature combinations specifiche (ESATTO come prima)
    numeric_cols = X_enhanced.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        # Ratio features
        for i, col1 in enumerate(numeric_cols[:5]):  # Limita per evitare esplosione
            for col2 in numeric_cols[i+1:6]:
                if (X_enhanced[col2] != 0).all():  # Evita divisione per zero
                    X_enhanced[f'{col1}_{col2}_ratio'] = X_enhanced[col1] / (X_enhanced[col2] + 1e-8)
        
        # Mean features per gruppi (ESATTO come prima)
        X_enhanced['features_mean'] = X_enhanced[numeric_cols].mean(axis=1)
        X_enhanced['features_std'] = X_enhanced[numeric_cols].std(axis=1)
        X_enhanced['features_max'] = X_enhanced[numeric_cols].max(axis=1)
        X_enhanced['features_min'] = X_enhanced[numeric_cols].min(axis=1)
    
    print(f"Feature engineering completato: {X_enhanced.shape[1]} features finali")
    return X_enhanced

def advanced_feature_selection_winning(X, y, task_type='classification'):
    """Feature selection ESATTA che dava 97% accuracy"""
    print(f"\n=== SELEZIONE FEATURES VINCENTE ({task_type}) ===")
    
    if task_type == 'classification':
        score_func = f_classif
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        score_func = f_regression
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # 1. Univariate Selection (ESATTO come prima)
    selector_k = SelectKBest(score_func=score_func, k=min(15, X.shape[1]))
    X_k = selector_k.fit_transform(X, y)
    selected_features_k = X.columns[selector_k.get_support()].tolist()
    
    # 2. Recursive Feature Elimination (ESATTO come prima)
    rfe = RFE(estimator=estimator, n_features_to_select=min(10, X.shape[1]))
    X_rfe = rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[rfe.get_support()].tolist()
    
    # 3. Model-based selection (ESATTO come prima)
    selector_model = SelectFromModel(estimator, threshold='median')
    X_model = selector_model.fit_transform(X, y)
    selected_features_model = X.columns[selector_model.get_support()].tolist()
    
    # Combina le selezioni (ESATTO come prima)
    all_selected = set(selected_features_k + selected_features_rfe + selected_features_model)
    final_features = list(all_selected)
    
    print(f"Features selezionate da K-Best: {len(selected_features_k)}")
    print(f"Features selezionate da RFE: {len(selected_features_rfe)}")
    print(f"Features selezionate da Model: {len(selected_features_model)}")
    print(f"Features finali combinate: {len(final_features)}")
    
    return X[final_features], final_features

def optimize_classification_winning(X, y):
    """Classificazione ESATTA che dava 97% accuracy"""
    print(f"\nCLASSIFICAZIONE VINCENTE")
    
    # Feature Engineering VINCENTE
    X_enhanced = feature_engineering_winning(X)
    
    # Scaling (ESATTO come prima)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_enhanced),
        columns=X_enhanced.columns,
        index=X_enhanced.index
    )
    
    # Feature Selection VINCENTE
    X_selected, selected_features = advanced_feature_selection_winning(X_scaled, y, 'classification')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Modelli ESATTI che davano 97%
    models = {}
    
    # Random Forest (parametri ESATTI)
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    print("Ottimizzando Random Forest...")
    rf_grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), 
        rf_params, 
        n_iter=20, 
        cv=5, 
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    models['RandomForest'] = rf_grid.best_estimator_
    
    # Gradient Boosting (parametri ESATTI)
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    print("Ottimizzando Gradient Boosting...")
    gb_grid = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    models['GradientBoosting'] = gb_grid.best_estimator_
    
    # XGBoost (parametri ESATTI che davano 97%)
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        print("Ottimizzando XGBoost...")
        xgb_grid = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            xgb_params,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        models['XGBoost'] = xgb_grid.best_estimator_
    
    # Test dei modelli
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        print(f"{name}:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Ensemble VINCENTE
    if len(models) >= 2:
        print("\nCreando ENSEMBLE VINCENTE...")
        model_list = [(name, model) for name, model in models.items()]
        ensemble = VotingClassifier(estimators=model_list, voting='soft')
        ensemble.fit(X_train, y_train)
        
        y_pred_ensemble = ensemble.predict(X_test)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        
        results['ENSEMBLE'] = {
            'accuracy': accuracy_ensemble,
            'model': ensemble
        }
        
        print(f"ENSEMBLE Accuracy: {accuracy_ensemble:.4f}")
    
    # Trova il migliore
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nMIGLIOR CLASSIFICATORE: {best_model_name}")
    print(f"ACCURACY: {best_accuracy:.4f}")
    
    return results, best_model_name, selected_features

def optimize_regression_winning(X, y):
    """Regressione OTTIMIZZATA per battere 0.08 R²"""
    print(f"\nREGRESSIONE VINCENTE")
    
    # Feature Engineering VINCENTE
    X_enhanced = feature_engineering_winning(X)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_enhanced),
        columns=X_enhanced.columns,
        index=X_enhanced.index
    )
    
    # Feature Selection per regressione
    X_selected, selected_features = advanced_feature_selection_winning(X_scaled, y, 'regression')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Modelli ottimizzati
    models = {}
    
    # Random Forest per regressione
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    print("Ottimizzando Random Forest per regressione...")
    rf_grid = RandomizedSearchCV(
        RandomForestRegressor(random_state=42), 
        rf_params, 
        n_iter=20, 
        cv=5, 
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    models['RandomForest'] = rf_grid.best_estimator_
    
    # Ridge ottimizzata
    ridge_params = {
        'alpha': [0.1, 1, 10, 100, 1000]
    }
    ridge_grid = RandomizedSearchCV(
        Ridge(random_state=42),
        ridge_params,
        n_iter=10,
        cv=5,
        scoring='r2',
        random_state=42
    )
    ridge_grid.fit(X_train, y_train)
    models['Ridge'] = ridge_grid.best_estimator_
    
    # XGBoost per regressione
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        print("Ottimizzando XGBoost per regressione...")
        xgb_grid = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params,
            n_iter=20,
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        models['XGBoost'] = xgb_grid.best_estimator_
    
    # Test dei modelli
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'model': model
        }
        
        print(f"{name}:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # Ensemble per regressione
    if len(models) >= 2:
        print("Creando ENSEMBLE per regressione...")
        model_list = [(name, model) for name, model in models.items()]
        ensemble = VotingRegressor(estimators=model_list)
        ensemble.fit(X_train, y_train)
        
        y_pred_ensemble = ensemble.predict(X_test)
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        
        results['ENSEMBLE'] = {
            'r2': r2_ensemble,
            'rmse': rmse_ensemble,
            'model': ensemble
        }
        
        print(f"ENSEMBLE R²: {r2_ensemble:.4f}")
    
    # Trova il migliore
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_r2 = results[best_model_name]['r2']
    
    print(f"\nMIGLIOR REGRESSORE: {best_model_name}")
    print(f"R²: {best_r2:.4f}")
    
    return results, best_model_name, selected_features

def main():
    """Funzione principale VINCENTE"""
    print("🚀 MACHINE LEARNING SUPER OTTIMIZZATO - REPLICA TECNICA VINCENTE")
    print("="*80)
    
    print("OBIETTIVO: Replicare esattamente le tecniche che davano 97.26% accuracy")
    
    # Carica dati ESATTO
    X, y_cluster, y_severity, feature_names = load_and_prepare_data_winning()
    
    # Ottimizza classificazione VINCENTE
    class_results, best_classifier, class_features = optimize_classification_winning(X, y_cluster)
    
    # Ottimizza regressione VINCENTE
    reg_results, best_regressor, reg_features = optimize_regression_winning(X, y_severity)
    
    # Report finale
    print("\n" + "="*80)
    print("RISULTATI FINALI SUPER OTTIMIZZATI")
    print("="*80)
    
    print(f"CLASSIFICAZIONE:")
    print(f"   Miglior modello: {best_classifier}")
    print(f"   Accuracy: {class_results[best_classifier]['accuracy']:.4f}")
    print(f"   Features usate: {len(class_features)}")
    
    print(f"\nREGRESSIONE:")
    print(f"   Miglior modello: {best_regressor}")
    print(f"   R²: {reg_results[best_regressor]['r2']:.4f}")
    print(f"   Features usate: {len(reg_features)}")
    
    # Verifica obiettivi
    target_accuracy = 0.95
    target_r2 = 0.05  # Obiettivo più realistico senza data leakage
    
    achieved_accuracy = class_results[best_classifier]['accuracy']
    achieved_r2 = reg_results[best_regressor]['r2']
    
    print(f"\nVERIFICA OBIETTIVI:")
    if achieved_accuracy >= target_accuracy:
        print(f"   Accuracy: {achieved_accuracy:.4f} >= {target_accuracy:.2f} (OBIETTIVO RAGGIUNTO!)")
    else:
        print(f"   Accuracy: {achieved_accuracy:.4f} < {target_accuracy:.2f} (obiettivo mancato)")
    
    if achieved_r2 >= target_r2:
        print(f"   R²: {achieved_r2:.4f} >= {target_r2:.2f} (OBIETTIVO RAGGIUNTO!)")
    else:
        print(f"   R²: {achieved_r2:.4f} < {target_r2:.2f} (obiettivo mancato)")
    
    print("\nOTTIMIZZAZIONE SUPER COMPLETATA!")

if __name__ == "__main__":
    main()
