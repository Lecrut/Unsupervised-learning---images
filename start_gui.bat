@echo off
echo ========================================
echo   Art Restoration AI - GUI Launcher
echo ========================================
echo.

REM Sprawdź czy środowisko jest aktywne
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Aktywuje srodowisko wirtualne...
    call venv\Scripts\activate.bat
)

REM Sprawdź czy Streamlit jest zainstalowany
python -c "import streamlit" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Streamlit nie jest zainstalowany!
    echo [i] Instaluje Streamlit...
    pip install streamlit
)

echo.
echo [*] Uruchamiam GUI...
echo [i] Aplikacja otworzy sie w przegladarce
echo [i] Aby zatrzymac, nacisnij Ctrl+C
echo.

streamlit run app_gui\app.py

pause
