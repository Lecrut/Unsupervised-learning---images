# Unsupervised Learning - Images# Unsupervised Learning - Images# Unsupervised Learning - Images



System przetwarzania obrazów dzieł sztuki wykorzystujący uczenie nienadzorowane i architekturę DeepCluster.



## Architektura DeepClusterSystem przetwarzania obrazów dzieł sztuki wykorzystujący uczenie nienadzorowane i architekturę DeepCluster.Projekt zawierający kompletną implementację systemu przetwarzania obrazów dzieł sztuki z wykorzystaniem metod uczenia nienadzorowanego:



### Etap I - Uczenie Enkodera- **Autoencoder** + klasteryzacja przestrzeni latentnej

- IMG: przetwarzanie wejściowego obrazu

- DMG: generowanie zniekształconych wersji obrazu## Architektura- **Inpainting** - uzupełnianie uszkodzonych fragmentów (proste i nieregularne)

- EMC: ekstrakcja reprezentacji latentnej

- **Super-Resolution** - zwiększanie rozdzielczości obrazów

### Etap II - Klasteryzacja

- PCAModule: redukcja wymiarowościProjekt implementuje architekturę DeepCluster składającą się z trzech etapów:- **GUI** - interfejs Streamlit do demonstracji

- ClusA: przypisywanie klastrów (KMeans, DBSCAN, GMM, Spectral)



### Etap III - Inpainting

- IMP: uzupełnianie w latent space### Etap I - Uczenie Enkodera (EMC)## 🎯 Cele projektu

- DEC: dekodowanie do przestrzeni obrazu

- IMG: przetwarzanie wejściowego obrazu

## Struktura projektu

- DMG: generowanie zniekształconych wersji obrazuProjekt realizuje założenia dla ocen:

```

src/- EMC: ekstrakcja reprezentacji latentnej- **3.0:** ✅ Autoencoder + klasteryzacja + inpainting prostych masek

├── data/

│   ├── damages.py- Funkcje straty: rekonstrukcja + spójność cech (contrastive loss)- **4.0:** ✅ Rozszerzenie o moduł super-resolution

│   ├── sampling.py

│   ├── splitting.py- **5.0:** ✅ Inpainting nieregularnych uszkodzeń + GUI + kompletna analiza

│   └── augmentations.py

├── models/### Etap II - Klasteryzacja (PCA + ClusA)

│   ├── autoencoder.py

│   ├── deepcluster_modules.py    # IMG, DMG, EMC, PCA, ClusA, IMP, DEC- PCA: redukcja wymiarowości przestrzeni latentnej## 📁 Struktura projektu

│   ├── wrappers.py               # EncoderModel, ClusteringModel, CometModel

│   ├── inpainting_model.py- ClusA: przypisywanie klastrów (KMeans, DBSCAN, GMM, Spectral)

│   └── superres_model.py

└── utils/```

    ├── training.py

    ├── analysis.py### Etap III - Inpainting (IMP + DEC)Unsupervised-learning---images/

    ├── visualization.py

    ├── metrics.py- IMP: uzupełnianie uszkodzonych fragmentów w latent space├── main.ipynb              # 🚀 Główny notebook z eksperymentami

    └── local_logger.py

```- DEC: dekodowanie do przestrzeni obrazu├── src/                    # Kod źródłowy



## Instalacja│   ├── models/            



```bash## Struktura projektu│   │   ├── autoencoder.py          # Konwolucyjny autoencoder

pip install -r requirements.txt

```│   │   ├── inpainting_model.py     # U-Net, PartialConv, SimpleInpainting



Windows:```│   │   └── superres_model.py       # SuperRes, LightweightSR, ESPCN

```powershell

py -m venv venvprojekt/│   ├── data/              

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt├── data/│   │   ├── damages.py              # Generowanie uszkodzeń (maski, szum, linie)

```

├── src/│   │   ├── sampling.py             # Podział i próbkowanie danych

## Uruchomienie

│   ├── data/│   │   ├── splitting.py            # Train/val/test split, cross-validation

```bash

jupyter notebook main.ipynb│   │   ├── damages.py            # Generowanie uszkodzeń│   │   └── augmentations.py        # Augmentacje danych

```

│   │   ├── sampling.py           # Próbkowanie danych│   └── utils/             

lub GUI:

```bash│   │   ├── splitting.py          # Podział train/val/test│       ├── training.py             # Funkcje trenowania z early stopping

streamlit run app_gui/app.py

```│   │   └── augmentations.py      # Augmentacje│       ├── analysis.py             # Klasteryzacja (KMeans, DBSCAN, GMM)



## Konfiguracja│   ├── models/│       ├── visualization.py        # Wizualizacje (UMAP, rekonstrukcje)



Utwórz plik `.env`:│   │   ├── autoencoder.py        # ConvAutoencoder│       ├── metrics.py              # SSIM, PSNR, MSE, MAE

```env

COMET_API_KEY=twoj_klucz│   │   ├── deepcluster_modules.py # IMG, DMG, EMC, PCA, ClusA, IMP, DEC│       └── local_logger.py         # Logowanie lokalne

COMET_PROJECT_NAME=nazwa_projektu

COMET_WORKSPACE=workspace│   │   ├── inpainting_model.py   # U-Net, PartialConv├── app_gui/               

```

│   │   └── superres_model.py     # Super-resolution│   └── app.py                      # 🎨 GUI Streamlit

## Użycie

│   └── utils/├── local_logs/                     # Logi eksperymentów

### DeepCluster Pipeline

```python│       ├── training.py           # Funkcje trenowania├── data/                           # Dane (WikiArt)

from src.models import DeepClusterPipeline

│       ├── analysis.py           # Klasteryzacja i analiza├── requirements.txt                # Zależności

model = DeepClusterPipeline(latent_dim=128, n_clusters=10, damage_type='mixed')

outputs = model(images, return_all=True)│       ├── visualization.py      # Wizualizacje└── README.md                       # Ten plik

```

│       ├── metrics.py            # PSNR, SSIM, MSE```

### Klasy opakowujące

```python│       └── local_logger.py       # Logowanie lokalne

from src.models import EncoderModel, ClusteringModel, InpaintingModel, CometModel

├── app_gui/## Instalacja

encoder = EncoderModel(latent_dim=128, device='cuda')

latent_vectors = encoder.extract_from_dataloader(train_loader, max_samples=5000)│   └── app.py                    # GUI Streamlit



clustering = ClusteringModel(n_clusters=10, algorithm='kmeans', use_pca=True)├── local_logs/                   # Logi eksperymentów### 1. Klonowanie repozytorium

labels = clustering.fit_predict(latent_vectors)

├── main.ipynb                    # Główny notebook```bash

inpainter = InpaintingModel(latent_dim=128, n_clusters=10, device='cuda')

reconstructed, damaged = inpainter.inpaint(images, labels)└── requirements.txtgit clone <URL_REPOZYTORIUM>



logger = CometModel("experiment_name", use_comet=True, use_local=True)```cd Unsupervised-learning---images

logger.log_parameters({'latent_dim': 128})

logger.log_metrics({'psnr': 28.5}, step=1)```

logger.end()

```## Wymagane biblioteki



### Moduły podstawowe### 2. Utworzenie wirtualnego środowiska (ZALECANE)

```python

from src.models import EMC, DMG, IMP, DEC, PCAModule, ClusA- PyTorch >= 1.12



encoder = EMC(latent_dim=128)- torchvision#### Opcja A: Używając venv (Windows)

dmg = DMG(damage_type='simple')

impainter = IMP(latent_dim=128, n_clusters=10)- scikit-learn```powershell

decoder = DEC(latent_dim=128)

```- umap-learn# Tworzenie wirtualnego środowiska



## Funkcjonalności- scikit-imagepy -m venv venv



**Ocena 3.0:**- imageio

- Autoencoder z reprezentacją latentną

- Klasteryzacja KMeans- numpy# Aktywacja środowiska

- Inpainting prostych masek

- matplotlib.\venv\Scripts\Activate.ps1

**Ocena 4.0:**

- Super-resolution- seaborn

- Metryki PSNR, SSIM, MS-SSIM

- datasets (Hugging Face)# Instalacja zależności

**Ocena 5.0:**

- Inpainting nieregularnych uszkodzeń- comet_ml (opcjonalnie)python -m pip install --upgrade pip

- GUI Streamlit

- Analiza klastrów- python-dotenvpip install -r requirements.txt

- Porównanie algorytmów klasteryzacji

- Streamlit```

## Metryki



- PSNR > 25 dB: bardzo dobra jakość

- SSIM > 0.85: wysokie podobieństwo strukturalne## Instalacja#### Opcja B: Używając conda

- Silhouette Score > 0.4: dobra jakość klasteryzacji

```bash

## Dataset

```bashconda create -n unsupervised-learning python=3.9

WikiArt z Hugging Face (Artificio/WikiArt_Full lub huggan/wikiart)

pip install -r requirements.txtconda activate unsupervised-learning

```pip install -r requirements.txt

```

### Windows

```powershell#### Opcja C: Instalacja globalna (niezalecane)

py -m venv venv```bash

.\venv\Scripts\Activate.ps1pip install -r requirements.txt

pip install -r requirements.txt```

```

### 3. Uruchomienie

### Linux/Mac```bash

```bash# Upewnij się, że wirtualne środowisko jest aktywne (powinieneś widzieć (venv) w promcie)

python3 -m venv venvjupyter notebook main.ipynb

source venv/bin/activate```

pip install -r requirements.txtlub

``````bash

jupyter lab main.ipynb

## Uruchomienie```



### Notebook## 🔧 Opcje logowania

```bash

jupyter notebook main.ipynb### ⚡ Szybkie testowanie

```Dla szybkich testów zmień w notebooku:

```python

### GUIDATASET_SIZE = 'test'  # 100 obrazów, ~2 minuty trenowania

```bash```

streamlit run app_gui/app.py

```Inne opcje:

```python

lubDATASET_SIZE = 500     # ~5 minut trenowania

DATASET_SIZE = 1000    # ~10 minut trenowania  

```powershellDATASET_SIZE = 0.1     # 10% datasetu

.\start_gui.ps1DATASET_SIZE = 'full'  # Pełny dataset (długo!)

``````



## Konfiguracja### 📁 Bez zewnętrznych serwisów (domyślnie)

- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = True`

### Logowanie- Wszystkie wyniki zapisywane lokalnie w folderze `local_logs/`

- Wykresy i metryki automatycznie zapisywane jako pliki

Utwórz plik `.env` na podstawie `.env.template`:

### 📊 Z Comet ML (opcjonalnie)

```env1. Skopiuj plik `.env.template` jako `.env`:

COMET_API_KEY=twoj_klucz   ```bash

COMET_PROJECT_NAME=nazwa_projektu   copy .env.template .env

COMET_WORKSPACE=workspace   ```

```2. Edytuj plik `.env` i uzupełnij swoje dane:

   ```env

W notebooku ustaw:   COMET_API_KEY=twoj_api_key_z_comet_ml

```python   COMET_PROJECT_NAME=nazwa_projektu

USE_COMET = True          # Comet ML   COMET_WORKSPACE=twoj_workspace

USE_LOCAL_LOGGER = True   # Lokalny logger   ```

```3. Ustaw `USE_COMET = True` w notebooku

4. Rejestracja na [comet.ml](https://www.comet.ml) wymagana

### Split danych

**Ważne:** Plik `.env` jest automatycznie ignorowany przez git, więc twoje dane pozostają bezpieczne.

```python

USE_QUICK_SPLIT = True   # 5000 próbek, ~1h treningu### 📝 Tylko konsola

USE_QUICK_SPLIT = False  # pełny dataset, 3-6h- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = False`

```- Wyniki tylko w konsoli, bez zapisywania



### Typy uszkodzeń### 4. Deaktywacja środowiska (po zakończeniu pracy)

```powershell

```pythondeactivate

DAMAGE_TYPE = 'simple'     # Kwadratowe maski```

DAMAGE_TYPE = 'irregular'  # Linie, plamy, szum

DAMAGE_TYPE = 'mixed'      # Wszystkie typy## 🎯 Szybkie uruchomienie (Windows)

```

### Opcja 1: Użyj gotowego skryptu

## Funkcjonalności```cmd

# Kliknij dwukrotnie na activate.bat

### Podstawowe (ocena 3.0)# lub w terminalu:

- Autoencoder z reprezentacją latentną.\activate.bat

- Klasteryzacja KMeans```

- Inpainting prostych masek

### Opcja 2: PowerShell

### Rozszerzone (ocena 4.0)```powershell

- Super-resolution.\activate.ps1

- Metryki PSNR, SSIM, MS-SSIM```

- Porównanie modeli

### Opcja 3: Ręcznie

### Zaawansowane (ocena 5.0)```powershell

- Inpainting nieregularnych uszkodzeń.\venv\Scripts\Activate.ps1

- GUI Streamlitjupyter notebook main.ipynb

- Analiza klastrów```

- Porównanie algorytmów (KMeans, DBSCAN, GMM, Spectral)

- Wizualizacje UMAP/t-SNE## ⚠️ Ważne uwagi

- **Zawsze aktywuj wirtualne środowisko** przed pracą z projektem

## Użycie modułów- Jeśli widzisz `(venv)` na początku linii w terminalu - środowisko jest aktywne

- Na Windows używaj `py` zamiast `python` do uruchamiania Pythona

### DeepCluster Pipeline- Jeśli masz problemy z PowerShell, spróbuj uruchomić jako administrator

```python

from src.models import DeepClusterPipeline## Struktura projektu

```

model = DeepClusterPipeline(Unsupervised-learning---images/

    latent_dim=128,├── main.ipynb              # główny notebook z eksperymentem

    n_clusters=10,├── src/                    # kod źródłowy

    damage_type='mixed'│   ├── __init__.py        # inicjalizacja pakietu

)│   ├── models/            # modele PyTorch

│   │   ├── __init__.py   

# Forward pass│   │   └── autoencoder.py # ConvAutoencoder

outputs = model(images, return_all=True)│   ├── data/              # przetwarzanie danych

```│   │   ├── __init__.py   

│   │   └── damages.py     # funkcje niszczenia obrazów

### Poszczególne moduły│   └── utils/             # funkcje pomocnicze

```python│       ├── __init__.py   

from src.models import EMC, DMG, IMP, DEC, PCAModule, ClusA│       ├── training.py    # funkcje trenowania

│       ├── analysis.py    # analiza latentna + klasteryzacja

encoder = EMC(latent_dim=128)│       └── visualization.py # wizualizacje

dmg = DMG(damage_type='simple')├── requirements.txt        # lista zależności

impainter = IMP(latent_dim=128, n_clusters=10)├── README.md              # ten plik

decoder = DEC(latent_dim=128)├── setup.py               # plik instalacyjny pakietu

pca = PCAModule(n_components=50)├── .gitignore             # pliki ignorowane przez git

clusterer = ClusA(n_clusters=10, algorithm='kmeans')├── activate.bat           # skrypt aktywacji dla CMD (Windows)

```├── activate.ps1           # skrypt aktywacji dla PowerShell (Windows)

├── venv/                  # wirtualne środowisko (nie commitowane)

## Metryki├── data/                  # folder na dane (nie commitowane)

└── *.pth                  # zapisane modele (nie commitowane)

- PSNR > 25 dB: bardzo dobra jakość```

- SSIM > 0.85: bardzo dobra podobieństwo strukturalne

- Silhouette Score > 0.4: dobra jakość klasteryzacji## 📚 Opis modułów



## Dane### 🧠 `src/models/autoencoder.py`

- `ConvAutoencoder` - konwolucyjny autoencoder z:

Projekt używa datasetu WikiArt z Hugging Face:  - Encoder: 3 warstwy konwolucyjne + BatchNorm

- Artificio/WikiArt_Full  - Decoder: 3 warstwy transponowane konwolucyjne

- huggan/wikiart  - Metody: `encode()`, `decode()`, `reconstruct()`



## Licencja### � `src/data/sampling.py`

- `LimitedDataset` - ogranicza liczbę próbek  

Projekt edukacyjny dla celów akademickich.- `QuickTestDataset` - mały dataset do testów (50-100 próbek)

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