# MMHAi — Music & Mental Health Analysis

Analisi della relazione tra abitudini musicali e salute mentale basata sul dataset MXMH Survey. Il progetto include preprocessing, clustering, modelli di machine learning e un’analisi con Reti Bayesiane.

## Dataset
- MXMH Survey Results (Kaggle): https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results
- Posizionare il file originale in `data/mxmh_survey_results.csv`.

## Struttura del repository
```
MMHAi---Music-Mental-Health-Analysis/
├─ data/
│  ├─ mxmh_survey_results.csv      # Dato originale (input richiesto)
│  └─ mxmh_final.csv               # Dato preprocessato e clusterizzato (output)
├─ images/                         # Grafici generati dagli script
├─ src/                            # Codice sorgente
│  ├─ main_preprocessing.py        # Preprocessing principale + clustering salute mentale
│  ├─ preprocessing.py             # EDA e analisi approfondita dei cluster
│  ├─ machine_learning_SUPER_optimized.py  # Pipeline ML (classificazione e regressione)
│  ├─ final_classifier_optimization.py     # GridSearch/ensemble (opzionale)
│  ├─ bayesian_analysis.py         # Analisi con Rete Bayesiana (pgmpy)
│  ├─ diversity_score_final.py     # Calcolo “diversity score” di ascolto
│  ├─ simple_classifier.py         # Baseline classificazione
│  └─ simple_mental_health_prediction.py  # Baseline regressione
├─ requirements.txt                # Dipendenze core del progetto
└─ README.md
```

## Requisiti
- Windows + PowerShell
- Python 3.11+
- Consigliato: Visual Studio Code

## Setup manuale (Windows PowerShell)
```powershell
# Verifica Python
python --version

# Crea un ambiente virtuale (facoltativo ma consigliato)
python -m venv new_venv

# Attiva l’ambiente
.\new_venv\Scripts\Activate.ps1

# Aggiorna pip e installa le dipendenze core
python -m pip install --upgrade pip
pip install -r requirements.txt

# (Opzionale) Installa librerie aggiuntive usate da alcuni script
# Necessario per alcune analisi/statistiche
pip install scipy
# Modelli opzionali in alcuni script (se vuoi provarli)
pip install xgboost lightgbm
```

Dipendenze principali (vedi anche `requirements.txt`):
- Core: pandas, numpy, seaborn, matplotlib, scikit-learn
- Probabilistic Graphical Models: pgmpy, bnlearn
- Utility: tqdm
- Opzionali: scipy (per test statistici in alcune analisi), xgboost, lightgbm

## Come eseguire
1) Preprocessing e generazione dataset finale
```powershell
python -u src\main_preprocessing.py
# Input:  data\mxmh_survey_results.csv
# Output: data\mxmh_final.csv
```
2) Analisi esplorativa e cluster
```powershell
python -u src\preprocessing.py
# Legge data\mxmh_final.csv e salva grafici in images/
```
3) Modelli di Machine Learning (classificazione/regressione)
```powershell
python -u src\machine_learning_SUPER_optimized.py
# Salva risultati/grafici in images/
```
4) Analisi con Rete Bayesiana
```powershell
python -u src\bayesian_analysis.py
# Salva grafici: struttura, distribuzioni, inferenze in images/
```
5) Diversity score (esplorazione generi al di fuori delle preferenze)
```powershell
python -u src\diversity_score_final.py
```
6) Script baseline
```powershell
python -u src\simple_classifier.py
python -u src\simple_mental_health_prediction.py
```

## Output
- I grafici vengono salvati in `images/` (es. `machine_learning_results.png`, `bayesian_network_structure.png`, ecc.).
- I risultati preprocessati sono salvati in `data/mxmh_final.csv`.

## Risoluzione problemi
- “Dataset non trovato”: verificare che `data/mxmh_survey_results.csv` esista prima di lanciare `main_preprocessing.py`.
- “ImportError: xgboost/lightgbm”: sono opzionali. Installarli con `pip install xgboost lightgbm` o ignorare i messaggi di disponibilità.
- “ImportError: scipy”: alcune analisi (test t) lo richiedono. Installare con `pip install scipy`.
- Grafici non visualizzati: i grafici sono comunque salvati in `images/`. Chiudere le finestre per proseguire l’esecuzione.

## Licenza e crediti
- Dataset: MXMH Survey Results (Kaggle) — rispettare i termini del dataset.
- Codice: uso accademico/didattico.


