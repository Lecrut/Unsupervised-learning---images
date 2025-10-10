# Skrypt aktywacji środowiska dla PowerShell
Write-Host "Aktywowanie wirtualnego środowiska..." -ForegroundColor Green

try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "✅ Środowisko aktywne!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Możesz teraz uruchomić:" -ForegroundColor Yellow
    Write-Host "  jupyter notebook main.ipynb" -ForegroundColor Cyan
    Write-Host "  - lub -" -ForegroundColor Yellow
    Write-Host "  jupyter lab main.ipynb" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Aby deaktywować środowisko, wpisz: deactivate" -ForegroundColor Yellow
}
catch {
    Write-Host "❌ Błąd aktywacji środowiska. Sprawdź czy folder venv istnieje." -ForegroundColor Red
    Write-Host "Jeśli nie, uruchom: py -m venv venv" -ForegroundColor Yellow
}