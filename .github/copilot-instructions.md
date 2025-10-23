# Instrukcja projektu: Uczenie Nienadzorowane - Inpainting i Super-Resolution Dzieł Sztuki

## Cel projektu

Celem projektu jest zbudowanie kompletnego systemu przetwarzania obrazów dzieł sztuki z wykorzystaniem metod uczenia nienadzorowanego i samonadzorowanego. System powinien umożliwiać:

1. Budowę reprezentacji obrazów (embeddingów) za pomocą autoenkodera.
2. Klasteryzację obrazów w przestrzeni latentnej (grupowanie stylów / autorów).
3. Uzupełnianie uszkodzonych fragmentów obrazów (inpainting) przy użyciu metod neuronowych.
4. Zwiększanie rozdzielczości obrazów (super-resolution) z wykorzystaniem modeli neuronowych.

Projekt obejmuje trzy poziomy zaawansowania (zgodnie z kryteriami oceny):

* **3.0:** Autoenkoder + klasteryzacja + proste uszkodzenia (maski kwadratowe)
* **4.0:** Rozszerzenie o moduł super-resolution
* **5.0:** Rozszerzenie o uzupełnianie nieregularnych uszkodzeń (np. maski losowe, krawędzie, plamy)

---

## Wymagane biblioteki (z `requirements.txt`)

* **Deep Learning:** PyTorch (>=1.12), torchvision
* **Machine Learning:** scikit-learn (KMeans, DBSCAN, GaussianMixture, SpectralClustering)
* **Redukcja wymiarowości:** umap-learn, TSNE
* **Obliczenia naukowe:** numpy, scikit-image, imageio
* **Wizualizacja:** matplotlib, seaborn
* **Dataset:** Hugging Face datasets (WikiArt_Full lub huggan/wikiart)
* **Eksperymenty:** comet_ml, python-dotenv
* **GUI:** Streamlit lub Dash

---

## Struktura projektu

```
projekt/
├── data/                          # Dane (WikiArt)
├── src/
│   ├── data/
│   │   ├── damages.py             # Generowanie uszkodzeń obrazów
│   │   ├── sampling.py            # Podział danych
│   │   └── augmentations.py       # Augmentacja danych
│   ├── models/
│   │   ├── autoencoder.py         # Autoenkoder / VAE
│   │   ├── inpainting_model.py    # Model do uzupełniania obrazów
│   │   └── superres_model.py      # Model do super-resolution
│   └── utils/
│       ├── training.py            # Pętla treningowa
│       ├── analysis.py            # Analiza wyników i klasterów
│       ├── visualization.py       # Wizualizacje UMAP, klastrów, rekonstrukcji
│       ├── local_logger.py        # Logowanie lokalne
│       └── comet_logger.py        # Integracja z Comet ML
├── app_gui/                       # GUI (Streamlit / Dash)
├── local_logs/                    # Logi eksperymentów
├── main.ipynb                     # Notebook główny
└── requirements.txt
```

---

## Etapy projektu

### 1. Przygotowanie danych

* Pobierz zbór `Artificio/WikiArt_Full` lub `huggan/wikiart` z Hugging Face.
* Ustal podział na zbiory: **train**, **val**, **test**.
* Znormalizuj dane i przygotuj odpowiedni DataLoader.

---

### 2. Budowa reprezentacji (Autoencoder / VAE)

* Encoder powinien redukować wymiar obrazu (256x256x3) do przestrzeni latentnej (128-512D).
* Decoder rekonstruuje obraz z reprezentacji latentnej.
* Model może być konwolucyjny lub wariacyjny (VAE) — wybór pozostaje otwarty.
* Funkcja straty: kombinacja MSE, SSIM i/lub innych metryk podobieństwa.
* Dodaj regularizację (Dropout, BatchNorm) i mechanizmy wczesnego zatrzymania.

---

### 3. Klasteryzacja i analiza przestrzeni latentnej

* Wykorzystaj redukcję wymiarowości (UMAP / TSNE) dla wizualizacji.
* Zastosuj klasyczne metody klasteryzacji (np. KMeans, GaussianMixture, DBSCAN).
* Oceń jakość grupowania metrykami typu Silhouette Score.
* Wizualizuj wyniki, koloruj punkty wg klas stylu i uzyskanych klastrów.

---

### 4. Moduł inpaintingu

* Implementuj model potrafiący uzupełniać brakujące fragmenty obrazu.
* Dla poziomu 3.0 wystarczają maski prostokątne (do 1/16 powierzchni).
* Dla poziomu 5.0 dodaj generowanie **nieregularnych masek** (np. z szumem, losowymi krawędziami lub plamami).
* Model może być oparty o autoenkoder, U-Net, VAE lub inne architektury konwolucyjne.
* Do oceny użyj SSIM, PSNR i MSE.

---

### 5. Moduł super-resolution (dla oceny 4.0 i wyższej)

* Zaimplementuj sieć uczącą się zwiększania rozdzielczości obrazów (np. z 128x128 do 256x256).
* Możesz wykorzystać architekturę opartą na CNN, Residual Blocks lub innych popularnych technikach SR.
* Wykorzystaj pary (niska jakość → wysoka jakość) z WikiArt lub zsyntezowane downsamplowane dane.

---

### 6. Eksperymentowanie i śledzenie wyników

* Użyj **Comet ML** do logowania metryk, hiperparametrów i przykładowych obrazów.
* Dodatkowo zapisz wyniki lokalnie w `local_logs/` (JSON / CSV).
* Konfiguracja (np. optimizer, learning rate, wagi strat) powinna być zdefiniowana w pliku konfiguracyjnym lub `.env`, nie wpisana na sztywno w kodzie.

---

### 7. Interfejs graficzny (GUI)

* Przygotuj prosty interfejs (Streamlit lub Dash) umożliwiający:

  * wczytanie obrazu testowego,
  * wybór rodzaju uszkodzenia lub operacji (inpainting / super-resolution),
  * uruchomienie modelu i prezentację wyników na żywo.

---

## Metryki i ewaluacja

| Aspekt           | Metryka          | Cel                       |
| ---------------- | ---------------- | ------------------------- |
| Rekonstrukcja    | SSIM             | > 0.85                    |
| Rekonstrukcja    | PSNR             | > 25 dB                   |
| Klasteryzacja    | Silhouette Score | > 0.4                     |
| Super-resolution | PSNR, SSIM       | wzrost jakości vs wejście |

---

## Dodatkowe możliwości (dla chętnych)

* Wykorzystanie **VAE** lub **Attention U-Net**.
* Transfer learning z modeli ImageNetowych (np. ResNet encoder).
* Warunkowe inpainting (na podstawie klas lub stylów).
* Interpolacja w przestrzeni latentnej.
* Multi-scale learning (uczenie rekonstrukcji na różnych rozdzielczościach).

---

## Dostarczane elementy

1. **Kod źródłowy** w strukturze `src/`.
2. **Notebook analityczny** `main.ipynb` prezentujący:
   * proces uczenia,
   * analizę latent space,
   * klasteryzację,
   * przykłady rekonstrukcji i super-resolution.
3. **Aplikacja GUI** (Streamlit/Dash) do prezentacji działania modelu.
4. **Logi eksperymentów** (Comet ML + lokalne JSON).
5. **Raport z wynikami i wnioskami** (README lub sekcja w notebooku).

---

## Kryteria ocen

| Ocena   | Wymagania                                                                 |
| ------- | ------------------------------------------------------------------------- |
| **3.0** | Autoenkoder + klasteryzacja + inpainting prostych masek                   |
| **4.0** | Jak wyżej + moduł super-resolution                                        |
| **5.0** | Jak wyżej + inpainting nieregularnych uszkodzeń + GUI + kompletna analiza |

---

## Wskazówki i dobre praktyki

* Stosuj **early stopping**, **gradient clipping** i **harmonogram uczenia** dla stabilności.
* Eksperymentuj z **latent_dim**, **głębokością sieci** i **typami strat**.
* Wersjonuj eksperymenty (Comet ML, lokalne logi).
* Zadbaj o czytelne wizualizacje (krzywe strat, rekonstrukcje, UMAP, wyniki GUI).

Powodzenia w realizacji projektu! 🚀
