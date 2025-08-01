# MMHAi Project Setup Script
# Questo script configura automaticamente l'ambiente di sviluppo

Write-Host "=== MMHAi Project Setup ===" -ForegroundColor Green
Write-Host "Configurazione automatica dell'ambiente di sviluppo..." -ForegroundColor Yellow

# Controlla se Python è installato
Write-Host "`n1. Controllo installazione Python..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✓ Python trovato: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python non trovato! Installa Python da https://python.org" -ForegroundColor Red
    exit 1
}

# Rimuove l'ambiente virtuale esistente se presente
Write-Host "`n2. Pulizia ambiente precedente..." -ForegroundColor Cyan
if (Test-Path "new_venv") {
    Write-Host "   → Rimuovendo ambiente virtuale esistente..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "new_venv" -ErrorAction SilentlyContinue
}

# Crea nuovo ambiente virtuale
Write-Host "`n3. Creazione nuovo ambiente virtuale..." -ForegroundColor Cyan
python -m venv new_venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ Ambiente virtuale creato" -ForegroundColor Green
} else {
    Write-Host "   ❌ Errore nella creazione dell'ambiente virtuale" -ForegroundColor Red
    exit 1
}

# Attiva l'ambiente virtuale e installa le dipendenze
Write-Host "`n4. Installazione dipendenze..." -ForegroundColor Cyan
try {
    & ".\new_venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\new_venv\Scripts\python.exe" -m pip install -r requirements.txt
    Write-Host "   ✓ Dipendenze installate con successo" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Errore nell'installazione delle dipendenze" -ForegroundColor Red
    exit 1
}

# Verifica installazione
Write-Host "`n5. Verifica installazione..." -ForegroundColor Cyan
$testResult = & ".\new_venv\Scripts\python.exe" -c "import pandas, matplotlib, seaborn, numpy, sklearn; print('Tutte le librerie importate con successo!')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ $testResult" -ForegroundColor Green
} else {
    Write-Host "   ❌ Errore nella verifica: $testResult" -ForegroundColor Red
    exit 1
}

# Controllo dataset
Write-Host "`n6. Controllo dataset..." -ForegroundColor Cyan
if (Test-Path "data\mxmh_clean.csv") {
    $datasetInfo = & ".\new_venv\Scripts\python.exe" -c "import pandas as pd; df = pd.read_csv('data/mxmh_clean.csv'); print(f'Dataset: {df.shape[0]} righe, {df.shape[1]} colonne')"
    Write-Host "   ✓ $datasetInfo" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Dataset non trovato in data/mxmh_clean.csv" -ForegroundColor Yellow
    Write-Host "   → Assicurati di avere il file dataset nella cartella data/" -ForegroundColor Yellow
}

Write-Host "`n=== SETUP COMPLETATO ===" -ForegroundColor Green
Write-Host "`nPer eseguire l'analisi:" -ForegroundColor White
Write-Host "   .\new_venv\Scripts\python.exe -u src\preprocessing.py" -ForegroundColor Cyan
Write-Host "`nPer attivare manualmente l'ambiente virtuale:" -ForegroundColor White
Write-Host "   .\new_venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nPer disattivare l'ambiente virtuale:" -ForegroundColor White
Write-Host "   deactivate" -ForegroundColor Cyan
