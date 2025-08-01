@echo off
echo === MMHAi Project Setup ===
echo Configurazione automatica dell'ambiente di sviluppo...

echo.
echo 1. Controllo installazione Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    ❌ Python non trovato! Installa Python da https://python.org
    pause
    exit /b 1
)
echo    ✓ Python trovato

echo.
echo 2. Pulizia ambiente precedente...
if exist "new_venv" (
    echo    → Rimuovendo ambiente virtuale esistente...
    rmdir /s /q "new_venv" >nul 2>&1
)

echo.
echo 3. Creazione nuovo ambiente virtuale...
python -m venv new_venv
if %errorlevel% neq 0 (
    echo    ❌ Errore nella creazione dell'ambiente virtuale
    pause
    exit /b 1
)
echo    ✓ Ambiente virtuale creato

echo.
echo 4. Installazione dipendenze...
new_venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
new_venv\Scripts\python.exe -m pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo    ❌ Errore nell'installazione delle dipendenze
    pause
    exit /b 1
)
echo    ✓ Dipendenze installate con successo

echo.
echo 5. Verifica installazione...
new_venv\Scripts\python.exe -c "import pandas, matplotlib, seaborn, numpy, sklearn; print('Tutte le librerie importate con successo!')"
if %errorlevel% neq 0 (
    echo    ❌ Errore nella verifica
    pause
    exit /b 1
)

echo.
echo 6. Controllo dataset...
if exist "data\mxmh_clean.csv" (
    new_venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/mxmh_clean.csv'); print(f'Dataset: {df.shape[0]} righe, {df.shape[1]} colonne')"
    echo    ✓ Dataset trovato
) else (
    echo    ⚠️  Dataset non trovato in data\mxmh_clean.csv
    echo    → Assicurati di avere il file dataset nella cartella data\
)

echo.
echo === SETUP COMPLETATO ===
echo.
echo Per eseguire l'analisi:
echo    new_venv\Scripts\python.exe -u src\preprocessing.py
echo.
echo Oppure usa:
echo    run.bat
echo.
pause
