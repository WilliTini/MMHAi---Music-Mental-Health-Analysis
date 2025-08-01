@echo off
echo === MMHAi Data Analysis ===

if not exist "new_venv\Scripts\python.exe" (
    echo ❌ Ambiente virtuale non trovato!
    echo Esegui prima: setup.bat
    pause
    exit /b 1
)

if not exist "data\mxmh_clean.csv" (
    echo ❌ Dataset non trovato!
    echo Assicurati che il file data\mxmh_clean.csv esista
    pause
    exit /b 1
)

echo 🚀 Avvio analisi dati...
echo I grafici si apriranno in finestre separate
echo Chiudi ogni finestra per passare al grafico successivo
echo.

new_venv\Scripts\python.exe -u src\preprocessing.py

echo.
echo ✅ Analisi completata!
pause
