# Aktywuj środowisko Pythona
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:Path"

# Zainstaluj wymagane pakiety
python -m pip install -r requirements.txt

# Sprawdź czy CUDA jest dostępna
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"