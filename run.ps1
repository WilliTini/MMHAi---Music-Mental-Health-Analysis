# Run MMHAi Analysis
# Script per eseguire facilmente l'analisi

Write-Host "=== MMHAi Data Analysis ===" -ForegroundColor Green

# Controlla se l'ambiente virtuale esiste
if (-not (Test-Path "new_venv\Scripts\python.exe")) {
    Write-Host "❌ Ambiente virtuale non trovato!" -ForegroundColor Red
    Write-Host "Esegui prima: .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Controlla se il dataset esiste
if (-not (Test-Path "data\mxmh_clean.csv")) {
    Write-Host "❌ Dataset non trovato!" -ForegroundColor Red
    Write-Host "Assicurati che il file data\mxmh_clean.csv esista" -ForegroundColor Yellow
    exit 1
}

Write-Host "🚀 Avvio analisi dati..." -ForegroundColor Cyan
Write-Host "I grafici si apriranno in finestre separate" -ForegroundColor Yellow
Write-Host "Chiudi ogni finestra per passare al grafico successivo`n" -ForegroundColor Yellow

# Esegue l'analisi
& ".\new_venv\Scripts\python.exe" -u "src\preprocessing.py"

Write-Host "`n✅ Analisi completata!" -ForegroundColor Green
