# Art Restoration AI - GUI Launcher (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Art Restoration AI - GUI Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Sprawdź czy środowisko wirtualne istnieje
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[*] Aktywuję środowisko wirtualne..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "[!] Brak środowiska wirtualnego!" -ForegroundColor Red
    Write-Host "[i] Uruchom najpierw: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Sprawdź czy Streamlit jest zainstalowany
$streamlitInstalled = python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[!] Streamlit nie jest zainstalowany!" -ForegroundColor Red
    Write-Host "[i] Instaluję Streamlit..." -ForegroundColor Yellow
    pip install streamlit
}

# Sprawdź czy model istnieje
if (-not (Test-Path "autoencoder.pth")) {
    Write-Host ""
    Write-Host "[!] Brak wytrenowanego modelu!" -ForegroundColor Red
    Write-Host "[i] Uruchom najpierw main.ipynb żeby wytrenować model" -ForegroundColor Yellow
    Write-Host "[i] Lub użyj demo: python demo.py --mode autoencoder" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[*] Uruchamiam GUI..." -ForegroundColor Green
Write-Host "[i] Aplikacja otworzy się w przeglądarce" -ForegroundColor Cyan
Write-Host "[i] Aby zatrzymać, naciśnij Ctrl+C" -ForegroundColor Cyan
Write-Host ""

# Uruchom Streamlit
streamlit run app_gui\app.py
