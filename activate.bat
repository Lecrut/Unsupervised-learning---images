@echo off
echo Aktywowanie wirtualnego środowiska...
call .\venv\Scripts\activate.bat
echo Środowisko aktywne! Możesz teraz uruchomić:
echo   jupyter notebook main.ipynb
echo   - lub -
echo   jupyter lab main.ipynb
echo.
echo Aby deaktywować środowisko, wpisz: deactivate
cmd /k