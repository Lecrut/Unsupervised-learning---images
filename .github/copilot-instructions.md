# Instrukcja projektu: Uczenie Nienadzorowane – Klasteryzacja, Inpainting i Super-Resolution Dzieł Sztuki

## Cel projektu

Celem projektu jest stworzenie kompletnego systemu przetwarzania obrazów dzieł sztuki z wykorzystaniem metod **uczenia nienadzorowanego** i **samouczącego (self-supervised)**, który umożliwia:

1. Budowę reprezentacji (embeddingu) obrazów za pomocą autoenkodera.  
2. Klasteryzację tych reprezentacji w przestrzeni latentnej w celu grupowania stylów lub autorów.  
3. Generowanie symulowanych uszkodzeń obrazów.  
4. Odtwarzanie (inpainting) uszkodzonych fragmentów obrazów przy użyciu metod neuronowych.  
5. (Opcjonalnie) zwiększanie rozdzielczości obrazów przy użyciu modelu super-resolution.

---

## Schemat przepływu danych

```
                         ┌──────────────────────────┐
                         │     Obraz nieuszkodzony  │
                         └─────────────┬────────────┘
                                       │
               ┌───────────────────────┼────────────────────────┐
               │                                                │
               ▼                                                ▼
     ┌──────────────────────┐                       ┌──────────────────────┐
     │       Encoder         │                       │     Generator        │
     │ (reprezentacja obrazu)│                       │     uszkodzeń        │
     └─────────────┬─────────┘                       └─────────────┬────────┘
                   │                                               │
                   ▼                                               ▼
     ┌──────────────────────┐                        ┌──────────────────────┐
     │ Reprezentacja latentna│                       │    Obraz uszkodzony  │
     └─────────────┬─────────┘                       └─────────┬────────────┘
                   │                                           │
                   ▼                                           │
     ┌──────────────────────┐                                  │
     │     Klasteryzacja     │                                 │
     └─────────────┬─────────┘                                 │
                   ▼                                           │
             ┌──────────────┐                                  │
             │    Klastry    │<──────────────┐                 │
             └──────────────┘               │                  │
                                            ▼                  ▼
                                 ┌──────────────────────────────┐
                                 │        Inpainting             │
                                 │ (klastry + obraz uszkodzony) │
                                 └──────────────┬───────────────┘
                                                ▼
                                      ┌──────────────────┐
                                      │     Dekoder      │
                                      └────────┬─────────┘
                                               ▼
                                  ┌───────────────────────────┐
                                  │  Obraz zrekonstruowany   │
                                  └───────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────────┐
                               │     [Super-Resolution model]    │
                               │ (poprawa jakości / rozdzielczości) │
                               └─────────────────┬─────────────────┘
                                                 ▼
                                   ┌──────────────────────────────┐
                                   │  Obraz końcowy wysokiej jakości │
                                   └──────────────────────────────┘

```

---

## Zakres projektu

Projekt realizuje następujące etapy:

### 1. Przygotowanie danych

- Pobierz zbiór `Artificio/WikiArt_Full` lub `huggan/wikiart` z platformy **Hugging Face**.  
- Podziel dane na zbiory: **train**, **val**, **test**.  
- Znormalizuj obrazy do formatu 256×256×3 i przygotuj `DataLoader`.

---

### 2. Generowanie uszkodzeń

- W pliku `damages.py` zaimplementuj funkcje do generowania:
  - prostych masek kwadratowych (do 1/16 obrazu) — wersja podstawowa,
  - masek nieregularnych (plamy, krawędzie, szum perlinowski) — wersja rozszerzona.  
- Wynik: **obraz uszkodzony** i odpowiadająca mu **maska uszkodzeń**.

---

### 3. Budowa reprezentacji (Encoder)

- Encoder przekształca obraz w reprezentację latentną o wymiarze 128–512.  
- Można wykorzystać klasyczny **autoenkoder konwolucyjny** lub **wariacyjny (VAE)**.  
- Funkcja straty: kombinacja MSE + SSIM.  
- Dodaj regularizację (Dropout, BatchNorm, Early Stopping).  

---

### 4. Klasteryzacja przestrzeni latentnej

- Wykorzystaj metody: **KMeans**, **GaussianMixture**, **DBSCAN**, **SpectralClustering**.  
- Wizualizuj wyniki w 2D przy pomocy **UMAP** lub **t-SNE**.  
- Oceń jakość grupowania metrykami typu **Silhouette Score**.  
- Porównaj klastry z metadanymi (np. styl, autor).

---

### 5. Inpainting

- Model inpaintingu przyjmuje **obraz uszkodzony** oraz (opcjonalnie) informację o **klastrze**.  
- Architektura: U-Net / Autoencoder / VAE.  
- Celem jest uzupełnienie brakujących fragmentów obrazu.  
- Funkcja straty: MSE, L1, perceptual loss, SSIM.  
- Ocena jakości: **SSIM**, **PSNR**, **MSE**.  

---

### 6. Dekoder i rekonstrukcja

- Dekoder rekonstruuje pełny obraz z przestrzeni latentnej po inpaintingu.  
- Porównaj wynik z oryginałem, zapisz przykładowe rekonstrukcje.  

---

### 7. Super-Resolution (opcjonalnie)

- Model SR poprawia rozdzielczość i jakość zrekonstruowanych obrazów.  
- Może być oparty o:
  - CNN z blokami rezydualnymi,
  - SRResNet, ESRGAN lub własną architekturę.  
- Trenuj na parach (obraz niskiej jakości → obraz oryginalny).  
- Metryki: **PSNR**, **SSIM**, **LPIPS**.  

---

### 8. Ewaluacja i wizualizacja

- Przedstaw wyniki klasteryzacji, inpaintingu i super-resolution na wizualizacjach 2D (UMAP, t-SNE).  
- Wizualizuj przykładowe:
  - obrazy oryginalne,  
  - obrazy uszkodzone,  
  - obrazy po rekonstrukcji i SR.  
- Oblicz średnie wartości SSIM/PSNR dla zbioru testowego.  

---

### 9. Interfejs graficzny (GUI)

Zaprojektuj prosty interfejs (Streamlit / Dash), który umożliwia:

- wczytanie obrazu testowego,  
- wybór rodzaju uszkodzenia (kwadratowe / nieregularne),  
- uruchomienie modelu inpaintingu lub super-resolution,  
- prezentację wynikowego obrazu w aplikacji.  

---

## Struktura projektu

```
projekt/
├── data/                          
├── src/
│   ├── data/
│   │   ├── damages.py             
│   │   └── sampling.py            
│   ├── models/
│   │   ├── encoder.py             
│   │   ├── decoder.py             
│   │   ├── clustering.py          
│   │   ├── inpainting_model.py    
│   │   └── superres_model.py      
│   └── utils/
│       ├── training.py            
│       ├── analysis.py            
│       ├── visualization.py       
│       └── comet_logger.py        
├── app_gui/                       
├── local_logs/                    
├── main.ipynb                     
└── requirements.txt
```

---

## Wymagane biblioteki

- **PyTorch** (>=1.12), **torchvision**  
- **scikit-learn** (KMeans, DBSCAN, GaussianMixture, SpectralClustering)  
- **umap-learn**, **scikit-image**, **imageio**, **numpy**  
- **matplotlib**, **seaborn**  
- **datasets** z Hugging Face  
- **comet_ml**, **python-dotenv**  
- **Streamlit** lub **Dash**  

---

## Kryteria ocen

| Ocena | Wymagania |
|:------|:-----------|
| **3.0** | Autoenkoder + klasteryzacja + inpainting prostych masek |
| **4.0** | Jak wyżej + super-resolution |
| **5.0** | Jak wyżej + inpainting nieregularnych uszkodzeń + GUI + analiza klastrów |

---

## Dobre praktyki i zasady kodowania

- Kod powinien być **czysty, spójny i czytelny**, zgodny z zasadami PEP8.  
- **Nie dodawaj zbędnych komentarzy** kopiowanych z dokumentacji — komentarze mają objaśniać logikę.  
- **Nazwy funkcji i klas** powinny jednoznacznie określać ich działanie (np. `generate_damage_mask`, `train_autoencoder`, `visualize_clusters`).  
- **Nie twórz dodatkowych plików podsumowujących** (np. `Updates.md`, `Summary.txt`) – cała dokumentacja powinna znajdować się w `README.md` i `main.ipynb`.  
- Każda funkcja powinna mieć jedną, dobrze zdefiniowaną odpowiedzialność.  
- Loguj eksperymenty w Comet ML i lokalnie w `local_logs/`.  
- Zadbaj o czytelne wizualizacje i opisy osi w wykresach.  

---

