# Unsupervised Learning - Images

Projekt zawierający implementację autoencodera z klasteryzacją i inpaintingiem obrazów.

## Instalacja

### 1. Klonowanie repozytorium
```bash
git clone <URL_REPOZYTORIUM>
cd Unsupervised-learning---images
```

### 2. Utworzenie wirtualnego środowiska (ZALECANE)

#### Opcja A: Używając venv (Windows)
```powershell
# Tworzenie wirtualnego środowiska
py -m venv venv

# Aktywacja środowiska
.\venv\Scripts\Activate.ps1

# Instalacja zależności
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Opcja B: Używając conda
```bash
conda create -n unsupervised-learning python=3.9
conda activate unsupervised-learning
pip install -r requirements.txt
```

#### Opcja C: Instalacja globalna (niezalecane)
```bash
pip install -r requirements.txt
```

### 3. Uruchomienie
```bash
# Upewnij się, że wirtualne środowisko jest aktywne (powinieneś widzieć (venv) w promcie)
jupyter notebook main.ipynb
```
lub
```bash
jupyter lab main.ipynb
```

## 🔧 Opcje logowania

### ⚡ Szybkie testowanie
Dla szybkich testów zmień w notebooku:
```python
DATASET_SIZE = 'test'  # 100 obrazów, ~2 minuty trenowania
```

Inne opcje:
```python
DATASET_SIZE = 500     # ~5 minut trenowania
DATASET_SIZE = 1000    # ~10 minut trenowania  
DATASET_SIZE = 0.1     # 10% datasetu
DATASET_SIZE = 'full'  # Pełny dataset (długo!)
```

### 📁 Bez zewnętrznych serwisów (domyślnie)
- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = True`
- Wszystkie wyniki zapisywane lokalnie w folderze `local_logs/`
- Wykresy i metryki automatycznie zapisywane jako pliki

### 📊 Z Comet ML (opcjonalnie)
1. Skopiuj plik `.env.template` jako `.env`:
   ```bash
   copy .env.template .env
   ```
2. Edytuj plik `.env` i uzupełnij swoje dane:
   ```env
   COMET_API_KEY=twoj_api_key_z_comet_ml
   COMET_PROJECT_NAME=nazwa_projektu
   COMET_WORKSPACE=twoj_workspace
   ```
3. Ustaw `USE_COMET = True` w notebooku
4. Rejestracja na [comet.ml](https://www.comet.ml) wymagana

**Ważne:** Plik `.env` jest automatycznie ignorowany przez git, więc twoje dane pozostają bezpieczne.

### 📝 Tylko konsola
- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = False`
- Wyniki tylko w konsoli, bez zapisywania

### 4. Deaktywacja środowiska (po zakończeniu pracy)
```powershell
deactivate
```

## 🎯 Szybkie uruchomienie (Windows)

### Opcja 1: Użyj gotowego skryptu
```cmd
# Kliknij dwukrotnie na activate.bat
# lub w terminalu:
.\activate.bat
```

### Opcja 2: PowerShell
```powershell
.\activate.ps1
```

### Opcja 3: Ręcznie
```powershell
.\venv\Scripts\Activate.ps1
jupyter notebook main.ipynb
```

## ⚠️ Ważne uwagi
- **Zawsze aktywuj wirtualne środowisko** przed pracą z projektem
- Jeśli widzisz `(venv)` na początku linii w terminalu - środowisko jest aktywne
- Na Windows używaj `py` zamiast `python` do uruchamiania Pythona
- Jeśli masz problemy z PowerShell, spróbuj uruchomić jako administrator

## Struktura projektu
```
Unsupervised-learning---images/
├── main.ipynb              # główny notebook z eksperymentem
├── src/                    # kod źródłowy
│   ├── __init__.py        # inicjalizacja pakietu
│   ├── models/            # modele PyTorch
│   │   ├── __init__.py   
│   │   └── autoencoder.py # ConvAutoencoder
│   ├── data/              # przetwarzanie danych
│   │   ├── __init__.py   
│   │   └── damages.py     # funkcje niszczenia obrazów
│   └── utils/             # funkcje pomocnicze
│       ├── __init__.py   
│       ├── training.py    # funkcje trenowania
│       ├── analysis.py    # analiza latentna + klasteryzacja
│       └── visualization.py # wizualizacje
├── requirements.txt        # lista zależności
├── README.md              # ten plik
├── setup.py               # plik instalacyjny pakietu
├── .gitignore             # pliki ignorowane przez git
├── activate.bat           # skrypt aktywacji dla CMD (Windows)
├── activate.ps1           # skrypt aktywacji dla PowerShell (Windows)
├── venv/                  # wirtualne środowisko (nie commitowane)
├── data/                  # folder na dane (nie commitowane)
└── *.pth                  # zapisane modele (nie commitowane)
```

## 📚 Opis modułów

### 🧠 `src/models/autoencoder.py`
- `ConvAutoencoder` - konwolucyjny autoencoder z:
  - Encoder: 3 warstwy konwolucyjne + BatchNorm
  - Decoder: 3 warstwy transponowane konwolucyjne
  - Metody: `encode()`, `decode()`, `reconstruct()`

### � `src/data/sampling.py`
- `LimitedDataset` - ogranicza liczbę próbek  
- `QuickTestDataset` - mały dataset do testów (50-100 próbek)
- `StratifiedLimitedDataset` - zachowuje proporcje klas
- `sample_dataset()` - różne metody próbkowania
- Obsługa: pełny dataset, konkretna liczba, procent

### �💥 `src/data/damages.py`
- `random_mask()` - losowe białe plamy
- `rectangular_mask()` - prostokątne maski
- `noise_mask()` - szum w pikselach
- `line_damage()` - losowe linie
- `circular_mask()` - okrągłe maski
- `MaskedDataset` - dataset z losowymi uszkodzeniami
- `MultiDamageDataset` - kombinacja wielu uszkodzeń

### 🏃‍♂️ `src/utils/training.py`
- `train_autoencoder()` - podstawowe trenowanie
- `validate_autoencoder()` - walidacja modelu
- `train_with_validation()` - trenowanie z early stopping

### 🔍 `src/utils/analysis.py`
- `extract_latent_vectors()` - ekstrakcja reprezentacji
- `cluster_latent_space()` - KMeans/Spectral/DBSCAN
- `reduce_dimensionality()` - UMAP/t-SNE/PCA
- `cluster_and_visualize()` - kompletna analiza

### 📊 `src/utils/visualization.py`
- `visualize_reconstructions()` - porównanie przed/po
- `plot_training_history()` - wykresy strat
- `visualize_latent_space_2d()` - embedding 2D
- `plot_cluster_analysis()` - szczegółowa analiza klastrów
- `compare_damage_types()` - porównanie typów uszkodzeń
```
Unsupervised-learning---images/
├── main.ipynb              # główny notebook z eksperymentem
├── src/                    # kod źródłowy
│   ├── __init__.py        # inicjalizacja pakietu
│   ├── models/            # modele PyTorch
│   │   ├── __init__.py   
│   │   └── autoencoder.py # ConvAutoencoder
│   ├── data/              # przetwarzanie danych
│   │   ├── __init__.py   
│   │   └── damages.py     # funkcje niszczenia obrazów
│   └── utils/             # funkcje pomocnicze
│       ├── __init__.py   
│       ├── training.py    # funkcje trenowania
│       ├── analysis.py    # analiza latentna + klasteryzacja
│       └── visualization.py # wizualizacje
├── requirements.txt        # lista zależności
├── README.md              # ten plik
├── setup.py               # plik instalacyjny pakietu
├── .gitignore             # pliki ignorowane przez git
├── activate.bat           # skrypt aktywacji dla CMD (Windows)
├── activate.ps1           # skrypt aktywacji dla PowerShell (Windows)
├── venv/                  # wirtualne środowisko (nie commitowane)
├── data/                  # folder na dane (nie commitowany)
└── *.pth                  # zapisane modele (nie commitowane)
```

## Wymagania systemowe
- Python 3.8+
- CUDA (opcjonalnie, dla GPU)

## Główne zależności
- **PyTorch** - framework do deep learning
- **torchvision** - narzędzia do przetwarzania obrazów
- **pytorch-msssim** - obliczanie SSIM (Structural Similarity Index) dla obrazów
- **scikit-learn** - algorytmy klasteryzacji (KMeans)
- **UMAP** - redukcja wymiarowości
- **Comet ML** - trackowanie eksperymentów (opcjonalne)
- **python-dotenv** - zarządzanie zmiennymi środowiskowymi
- **Jupyter** - środowisko notebookowe