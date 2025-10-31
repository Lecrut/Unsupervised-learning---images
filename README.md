# Unsupervised Learning - Images# Unsupervised Learning - Images# Unsupervised Learning - Images# Unsupervised Learning - Images



System przetwarzania obrazów dzieł sztuki wykorzystujący uczenie nienadzorowane i architekturę DeepCluster.



## Architektura DeepClusterSystem przetwarzania obrazów dzieł sztuki wykorzystujący uczenie nienadzorowane i architekturę DeepCluster.



Projekt implementuje architekturę DeepCluster składającą się z trzech etapów:



### Etap I - Uczenie Enkodera (EMC)## Architektura DeepClusterSystem przetwarzania obrazów dzieł sztuki wykorzystujący uczenie nienadzorowane i architekturę DeepCluster.Projekt zawierający kompletną implementację systemu przetwarzania obrazów dzieł sztuki z wykorzystaniem metod uczenia nienadzorowanego:

- IMG: przetwarzanie wejściowego obrazu

- DMG: generowanie zniekształconych wersji obrazu

- EMC: ekstrakcja reprezentacji latentnej

- Funkcje straty: rekonstrukcja + spójność cech (contrastive loss)### Etap I - Uczenie Enkodera- **Autoencoder** + klasteryzacja przestrzeni latentnej



### Etap II - Klasteryzacja (PCA + ClusA)- IMG: przetwarzanie wejściowego obrazu

- PCAModule: redukcja wymiarowości przestrzeni latentnej

- ClusA: przypisywanie klastrów (KMeans, DBSCAN, GMM, Spectral)- DMG: generowanie zniekształconych wersji obrazu## Architektura- **Inpainting** - uzupełnianie uszkodzonych fragmentów (proste i nieregularne)



### Etap III - Inpainting (IMP + DEC)- EMC: ekstrakcja reprezentacji latentnej

- IMP: uzupełnianie uszkodzonych fragmentów w latent space

- DEC: dekodowanie do przestrzeni obrazu- **Super-Resolution** - zwiększanie rozdzielczości obrazów



## Cele projektu### Etap II - Klasteryzacja



Projekt realizuje założenia dla ocen:- PCAModule: redukcja wymiarowościProjekt implementuje architekturę DeepCluster składającą się z trzech etapów:- **GUI** - interfejs Streamlit do demonstracji

- **3.0:** Autoencoder + klasteryzacja + inpainting prostych masek

- **4.0:** Rozszerzenie o moduł super-resolution- ClusA: przypisywanie klastrów (KMeans, DBSCAN, GMM, Spectral)

- **5.0:** Inpainting nieregularnych uszkodzeń + GUI + kompletna analiza



## Struktura projektu

### Etap III - Inpainting

```

Unsupervised-learning---images/- IMP: uzupełnianie w latent space### Etap I - Uczenie Enkodera (EMC)## 🎯 Cele projektu

├── main.ipynb              # Główny notebook z eksperymentami

├── src/- DEC: dekodowanie do przestrzeni obrazu

│   ├── models/

│   │   ├── autoencoder.py          # Konwolucyjny autoencoder- IMG: przetwarzanie wejściowego obrazu

│   │   ├── deepcluster_modules.py  # IMG, DMG, EMC, PCA, ClusA, IMP, DEC

│   │   ├── wrappers.py             # EncoderModel, ClusteringModel, CometModel## Struktura projektu

│   │   ├── inpainting_model.py     # U-Net, PartialConv, SimpleInpainting

│   │   └── superres_model.py       # SuperRes, LightweightSR, ESPCN- DMG: generowanie zniekształconych wersji obrazuProjekt realizuje założenia dla ocen:

│   ├── data/

│   │   ├── damages.py              # Generowanie uszkodzeń (maski, szum, linie)```

│   │   ├── sampling.py             # Podział i próbkowanie danych

│   │   ├── splitting.py            # Train/val/test split, cross-validationsrc/- EMC: ekstrakcja reprezentacji latentnej- **3.0:** ✅ Autoencoder + klasteryzacja + inpainting prostych masek

│   │   └── augmentations.py        # Augmentacje danych

│   └── utils/├── data/

│       ├── training.py             # Funkcje trenowania z early stopping

│       ├── analysis.py             # Klasteryzacja (KMeans, DBSCAN, GMM)│   ├── damages.py- Funkcje straty: rekonstrukcja + spójność cech (contrastive loss)- **4.0:** ✅ Rozszerzenie o moduł super-resolution

│       ├── visualization.py        # Wizualizacje (UMAP, rekonstrukcje)

│       ├── metrics.py              # SSIM, PSNR, MSE, MAE│   ├── sampling.py

│       └── local_logger.py         # Logowanie lokalne

├── app_gui/│   ├── splitting.py- **5.0:** ✅ Inpainting nieregularnych uszkodzeń + GUI + kompletna analiza

│   └── app.py                      # GUI Streamlit

├── local_logs/                     # Logi eksperymentów│   └── augmentations.py

├── data/                           # Dane (WikiArt)

├── requirements.txt                # Zależności├── models/### Etap II - Klasteryzacja (PCA + ClusA)

└── README.md                       # Ten plik

```│   ├── autoencoder.py



## Instalacja│   ├── deepcluster_modules.py    # IMG, DMG, EMC, PCA, ClusA, IMP, DEC- PCA: redukcja wymiarowości przestrzeni latentnej## 📁 Struktura projektu



### 1. Klonowanie repozytorium│   ├── wrappers.py               # EncoderModel, ClusteringModel, CometModel



```bash│   ├── inpainting_model.py- ClusA: przypisywanie klastrów (KMeans, DBSCAN, GMM, Spectral)

git clone <URL_REPOZYTORIUM>

cd Unsupervised-learning---images│   └── superres_model.py

```

└── utils/```

### 2. Utworzenie wirtualnego środowiska

    ├── training.py

#### Windows (PowerShell)

```powershell    ├── analysis.py### Etap III - Inpainting (IMP + DEC)Unsupervised-learning---images/

py -m venv venv

.\venv\Scripts\Activate.ps1    ├── visualization.py

python -m pip install --upgrade pip

pip install -r requirements.txt    ├── metrics.py- IMP: uzupełnianie uszkodzonych fragmentów w latent space├── main.ipynb              # 🚀 Główny notebook z eksperymentami

```

    └── local_logger.py

#### Linux/Mac

```bash```- DEC: dekodowanie do przestrzeni obrazu├── src/                    # Kod źródłowy

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt## Instalacja│   ├── models/            

```



### 3. Konfiguracja Comet ML (opcjonalnie)

```bash## Struktura projektu│   │   ├── autoencoder.py          # Konwolucyjny autoencoder

1. Skopiuj plik `.env.template` jako `.env`:

```bashpip install -r requirements.txt

copy .env.template .env  # Windows

cp .env.template .env    # Linux/Mac```│   │   ├── inpainting_model.py     # U-Net, PartialConv, SimpleInpainting

```



2. Edytuj plik `.env` i uzupełnij swoje dane:

```envWindows:```│   │   └── superres_model.py       # SuperRes, LightweightSR, ESPCN

COMET_API_KEY=twoj_api_key_z_comet_ml

COMET_PROJECT_NAME=nazwa_projektu```powershell

COMET_WORKSPACE=twoj_workspace

```py -m venv venvprojekt/│   ├── data/              



3. Ustaw `USE_COMET = True` w notebooku.\venv\Scripts\Activate.ps1



## Uruchomieniepip install -r requirements.txt├── data/│   │   ├── damages.py              # Generowanie uszkodzeń (maski, szum, linie)



### Notebook```

```bash

jupyter notebook main.ipynb├── src/│   │   ├── sampling.py             # Podział i próbkowanie danych

```

## Uruchomienie

### GUI

```bash│   ├── data/│   │   ├── splitting.py            # Train/val/test split, cross-validation

streamlit run app_gui/app.py

``````bash



## Opcje logowaniajupyter notebook main.ipynb│   │   ├── damages.py            # Generowanie uszkodzeń│   │   └── augmentations.py        # Augmentacje danych



### Bez zewnętrznych serwisów (domyślnie)```

- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = True`

- Wszystkie wyniki zapisywane lokalnie w folderze `local_logs/`│   │   ├── sampling.py           # Próbkowanie danych│   └── utils/             



### Z Comet ML (opcjonalnie)lub GUI:

- Ustaw `USE_COMET = True` i `USE_LOCAL_LOGGER = True`

- Wyniki logowane do Comet ML + lokalnie```bash│   │   ├── splitting.py          # Podział train/val/test│       ├── training.py             # Funkcje trenowania z early stopping



### Tylko konsolastreamlit run app_gui/app.py

- Ustaw `USE_COMET = False` i `USE_LOCAL_LOGGER = False`

- Wyniki tylko w konsoli```│   │   └── augmentations.py      # Augmentacje│       ├── analysis.py             # Klasteryzacja (KMeans, DBSCAN, GMM)



## Użycie modułów



### DeepCluster Pipeline## Konfiguracja│   ├── models/│       ├── visualization.py        # Wizualizacje (UMAP, rekonstrukcje)

```python

from src.models import DeepClusterPipeline



model = DeepClusterPipeline(Utwórz plik `.env`:│   │   ├── autoencoder.py        # ConvAutoencoder│       ├── metrics.py              # SSIM, PSNR, MSE, MAE

    latent_dim=128,

    n_clusters=10,```env

    damage_type='mixed'

)COMET_API_KEY=twoj_klucz│   │   ├── deepcluster_modules.py # IMG, DMG, EMC, PCA, ClusA, IMP, DEC│       └── local_logger.py         # Logowanie lokalne



outputs = model(images, return_all=True)COMET_PROJECT_NAME=nazwa_projektu

```

COMET_WORKSPACE=workspace│   │   ├── inpainting_model.py   # U-Net, PartialConv├── app_gui/               

### Klasy opakowujące

```python```

from src.models import EncoderModel, ClusteringModel, InpaintingModel, CometModel

│   │   └── superres_model.py     # Super-resolution│   └── app.py                      # 🎨 GUI Streamlit

# Encoder

encoder = EncoderModel(latent_dim=128, device='cuda')## Użycie

latent_vectors = encoder.extract_from_dataloader(train_loader, max_samples=5000)

│   └── utils/├── local_logs/                     # Logi eksperymentów

# Klasteryzacja

clustering = ClusteringModel(n_clusters=10, algorithm='kmeans', use_pca=True)### DeepCluster Pipeline

labels = clustering.fit_predict(latent_vectors)

```python│       ├── training.py           # Funkcje trenowania├── data/                           # Dane (WikiArt)

# Inpainting

inpainter = InpaintingModel(latent_dim=128, n_clusters=10, device='cuda')from src.models import DeepClusterPipeline

reconstructed, damaged = inpainter.inpaint(images, labels)

│       ├── analysis.py           # Klasteryzacja i analiza├── requirements.txt                # Zależności

# Logger

logger = CometModel("experiment_name", use_comet=True, use_local=True)model = DeepClusterPipeline(latent_dim=128, n_clusters=10, damage_type='mixed')

logger.log_parameters({'latent_dim': 128})

logger.log_metrics({'psnr': 28.5}, step=1)outputs = model(images, return_all=True)│       ├── visualization.py      # Wizualizacje└── README.md                       # Ten plik

logger.end()

``````



### Moduły podstawowe│       ├── metrics.py            # PSNR, SSIM, MSE```

```python

from src.models import EMC, DMG, IMP, DEC, PCAModule, ClusA### Klasy opakowujące



encoder = EMC(latent_dim=128)```python│       └── local_logger.py       # Logowanie lokalne

dmg = DMG(damage_type='simple')

impainter = IMP(latent_dim=128, n_clusters=10)from src.models import EncoderModel, ClusteringModel, InpaintingModel, CometModel

decoder = DEC(latent_dim=128)

pca = PCAModule(n_components=50)├── app_gui/## Instalacja

clusterer = ClusA(n_clusters=10, algorithm='kmeans')

```encoder = EncoderModel(latent_dim=128, device='cuda')



## Konfiguracja eksperymentulatent_vectors = encoder.extract_from_dataloader(train_loader, max_samples=5000)│   └── app.py                    # GUI Streamlit



### Split danych

```python

USE_QUICK_SPLIT = True   # 5000 próbek, ~1h treninguclustering = ClusteringModel(n_clusters=10, algorithm='kmeans', use_pca=True)├── local_logs/                   # Logi eksperymentów### 1. Klonowanie repozytorium

USE_QUICK_SPLIT = False  # pełny dataset, 3-6h

```labels = clustering.fit_predict(latent_vectors)



### Typy uszkodzeń├── main.ipynb                    # Główny notebook```bash

```python

DAMAGE_TYPE = 'simple'     # Kwadratowe maskiinpainter = InpaintingModel(latent_dim=128, n_clusters=10, device='cuda')

DAMAGE_TYPE = 'irregular'  # Linie, plamy, szum

DAMAGE_TYPE = 'mixed'      # Wszystkie typyreconstructed, damaged = inpainter.inpaint(images, labels)└── requirements.txtgit clone <URL_REPOZYTORIUM>

```



## Funkcjonalności

logger = CometModel("experiment_name", use_comet=True, use_local=True)```cd Unsupervised-learning---images

### Podstawowe (ocena 3.0)

- Autoencoder z reprezentacją latentnąlogger.log_parameters({'latent_dim': 128})

- Klasteryzacja KMeans

- Inpainting prostych maseklogger.log_metrics({'psnr': 28.5}, step=1)```



### Rozszerzone (ocena 4.0)logger.end()

- Super-resolution

- Metryki PSNR, SSIM, MS-SSIM```## Wymagane biblioteki

- Porównanie modeli



### Zaawansowane (ocena 5.0)

- Inpainting nieregularnych uszkodzeń### Moduły podstawowe### 2. Utworzenie wirtualnego środowiska (ZALECANE)

- GUI Streamlit

- Analiza klastrów```python

- Porównanie algorytmów (KMeans, DBSCAN, GMM, Spectral)

- Wizualizacje UMAP/t-SNEfrom src.models import EMC, DMG, IMP, DEC, PCAModule, ClusA- PyTorch >= 1.12



## Metryki



- PSNR > 25 dB: bardzo dobra jakośćencoder = EMC(latent_dim=128)- torchvision#### Opcja A: Używając venv (Windows)

- SSIM > 0.85: wysokie podobieństwo strukturalne

- Silhouette Score > 0.4: dobra jakość klasteryzacjidmg = DMG(damage_type='simple')



## Datasetimpainter = IMP(latent_dim=128, n_clusters=10)- scikit-learn```powershell



WikiArt z Hugging Face:decoder = DEC(latent_dim=128)

- Artificio/WikiArt_Full

- huggan/wikiart```- umap-learn# Tworzenie wirtualnego środowiska



## Wymagania systemowe



- Python 3.8+## Funkcjonalności- scikit-imagepy -m venv venv

- CUDA (opcjonalnie, dla GPU)



## Główne zależności

**Ocena 3.0:**- imageio

- PyTorch >= 1.12

- torchvision- Autoencoder z reprezentacją latentną

- scikit-learn

- umap-learn- Klasteryzacja KMeans- numpy# Aktywacja środowiska

- scikit-image

- matplotlib, seaborn- Inpainting prostych masek

- datasets (Hugging Face)

- comet_ml (opcjonalnie)- matplotlib.\venv\Scripts\Activate.ps1

- streamlit

**Ocena 4.0:**

## Licencja

- Super-resolution- seaborn

Projekt edukacyjny dla celów akademickich.

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