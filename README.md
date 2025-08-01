# MMHAi - Music & Mental Health Analysis

Questo progetto analizza la relazione tra abitudini musicali e salute mentale utilizzando Python, pandas, matplotlib e seaborn.

## 🚀 Setup Automatico

### Windows
1. **Prima esecuzione su una nuova macchina:**
   ```powershell
   .\setup.ps1
   ```
   Questo script:
   - Verifica che Python sia installato
   - Crea un ambiente virtuale
   - Installa tutte le dipendenze necessarie
   - Verifica che tutto funzioni correttamente

2. **Eseguire l'analisi:**
   ```powershell
   .\run.ps1
   ```

### Setup Manuale (se preferisci)
```powershell
# Crea ambiente virtuale
python -m venv new_venv

# Attiva ambiente virtuale
.\new_venv\Scripts\Activate.ps1

# Installa dipendenze
pip install -r requirements.txt

# Esegui analisi
python -u src\preprocessing.py
```

## 📁 Struttura del Progetto

```
MMHAi/
├── data/
│   └── mxmh_clean.csv          # Dataset pulito
├── src/
│   └── preprocessing.py        # Script principale di analisi
├── requirements.txt            # Dipendenze Python
├── setup.ps1                  # Script setup automatico
├── run.ps1                     # Script esecuzione rapida
└── README.md                   # Questa documentazione
```

## 📊 Grafici Generati

L'analisi produce 4 grafici interattivi:
1. **Distribuzione delle Età** - Istogramma con KDE
2. **Categorie di Ascolto** - Distribuzione delle ore di ascolto categorizzate
3. **Servizi di Streaming** - Distribuzione dei servizi più utilizzati
4. **Ore per Giorno** - Distribuzione delle ore di ascolto giornaliere

## 🔧 Requisiti di Sistema

- **Python 3.11+** - [Download da python.org](https://python.org)
- **PowerShell** (incluso in Windows)
- **Dataset** - File `data/mxmh_clean.csv` deve essere presente

## 📦 Dipendenze

- pandas 2.3.1
- matplotlib 3.10.5
- seaborn 0.13.2
- numpy 2.3.2
- scikit-learn 1.7.1

## 🐛 Risoluzione Problemi

### "Python non trovato"
- Installa Python da [python.org](https://python.org)
- Assicurati che Python sia nel PATH di sistema

### "Dataset non trovato"
- Verifica che il file `data/mxmh_clean.csv` esista
- Controlla che il percorso sia corretto

### "Errore PowerShell execution policy"
- Esegui: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Grafici non si aprono
- Il problema potrebbe essere con il backend di matplotlib
- I grafici dovrebbero aprirsi in finestre separate
- Chiudi ogni finestra per passare al grafico successivo

## 🔄 Aggiornamento

Per aggiornare l'ambiente:
```powershell
.\setup.ps1  # Reinstalla tutto da zero
```

## 👥 Contributori

- WilliTini - Sviluppatore principale
# Music & Mental Health AI

This project explores the correlation between music listening habits and mental health indicators (such as depression, anxiety, insomnia, and OCD) using the [MXMH Survey Dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results).  
It integrates exploratory data analysis, supervised learning, and symbolic reasoning to identify meaningful patterns and build interpretable models.

## 🎯 Objectives

- Analyze patterns in music preferences and mental health scores
- Train ML models to predict psychological well-being based on music habits
- Build a symbolic knowledge base to reason about mental health risk based on user profiles
- Compare machine learning predictions with rule-based reasoning

## 🧰 Technologies & Tools

- Python (pandas, seaborn, scikit-learn, Owlready2)
- Knowledge representation via OWL and rule-based reasoning
- MXMH Survey Dataset from Kaggle

## 📂 Project Structure

- `data/`: Original and cleaned versions of the dataset  
- `notebooks/`: EDA, ML models, and reasoning experiments  
- `kb/`: Ontologies, rules, and reasoning modules  
- `report/`: Final project documentation and analysis

## ✅ Status

- [x] Data cleaning and preprocessing  
- [x] Exploratory data analysis  
- [ ] Supervised learning models  
- [ ] Knowledge base and reasoning system  
- [ ] Evaluation and final report


