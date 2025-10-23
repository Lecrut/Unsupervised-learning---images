# 🎨 GUI - Streamlit App

Interfejs graficzny do demonstracji modeli inpainting i super-resolution.

## 📸 Zrzuty ekranu

### Tryb Inpainting
Uzupełnianie uszkodzonych fragmentów dzieł sztuki:
- Wczytanie obrazu
- Wybór typu uszkodzenia (plamy, linie, szum)
- Rekonstrukcja przez AI
- Porównanie przed/po

### Tryb Super-Resolution
Zwiększanie rozdzielczości obrazów:
- Symulacja niskiej rozdzielczości
- Zwiększanie przez model SR
- Porównanie z interpolacją bicubic

## 🚀 Uruchomienie

### Metoda 1: Bezpośrednio
```bash
# Z głównego folderu projektu
streamlit run app_gui/app.py
```

### Metoda 2: Z PowerShell
```powershell
# Aktywuj środowisko
.\venv\Scripts\Activate.ps1

# Uruchom aplikację
streamlit run app_gui/app.py
```

### Metoda 3: Ze skryptem
```bash
# Windows
start_gui.bat

# PowerShell
.\start_gui.ps1
```

## 📋 Wymagania

Aplikacja wymaga:
- ✅ Zainstalowanego `streamlit` (w requirements.txt)
- ✅ Wytrenowanego modelu `autoencoder.pth` w głównym folderze
- 📦 (Opcjonalnie) Modele inpainting i super-resolution

## 🎯 Funkcje

### Tryb Inpainting

**Typy uszkodzeń:**
- 🔲 Losowa plama - kwadratowa maska w losowym miejscu
- 📦 Prostokąt - maska w centrum obrazu
- 🌫️ Szum - losowe piksele zaszumione
- ➖ Linie - losowe linie przez obraz
- ⭕ Okrągła plama - okrągła maska w losowym miejscu

**Parametry konfigurowalne:**
- Rozmiar plamy/maski
- Poziom szumu
- Liczba linii
- Promień okręgu

**Modele:**
- Autoencoder (podstawowy) - wytrenowany model z main.ipynb
- U-Net (zaawansowany) - wymaga wytrenowania
- Simple Inpainting - wymaga wytrenowania

### Tryb Super-Resolution

**Funkcje:**
- Symulacja niskiej rozdzielczości (2x, 4x downscale)
- Zwiększanie przez model SR
- Porównanie z baseline (bicubic interpolation)

**Modele:**
- Standard SR - pełny model z residual blocks
- Lightweight SR - szybsza wersja
- ESPCN SR - najszybszy model

## 🔧 Konfiguracja

### Ścieżki do modeli
Edytuj w `app.py`:
```python
# Domyślne ścieżki
AUTOENCODER_PATH = "autoencoder.pth"
UNET_PATH = "unet_inpainting.pth"
SUPERRES_PATH = "superres.pth"
```

### Zmiana parametrów domyślnych
```python
# Domyślny rozmiar obrazu
DEFAULT_IMAGE_SIZE = (256, 256)

# Domyślny batch size
BATCH_SIZE = 1  # Dla GUI zawsze 1
```

## 💡 Wskazówki

### Optymalizacja wydajności
- GUI używa CPU - jeśli masz GPU, możesz zmienić device w kodzie
- Cache'owanie modeli (`@st.cache_resource`) przyśpiesza ponowne ładowanie
- Dla dużych obrazów może być wolno - ogranicz rozmiar

### Dodawanie własnych modeli
```python
@st.cache_resource
def load_my_model(model_path):
    model = MyModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
```

### Rozszerzanie GUI

**Dodaj nowy typ uszkodzenia:**
1. Dodaj funkcję w `src/data/damages.py`
2. Dodaj do listy w `app.py`:
```python
damage_type = st.sidebar.selectbox(
    "Typ uszkodzenia:",
    ["Losowa plama", "Prostokąt", "Twoje nowe uszkodzenie"]
)
```
3. Dodaj obsługę w `apply_damage()`

**Dodaj nowy model:**
1. Załaduj model w sekcji cache
2. Dodaj do selectboxa
3. Dodaj obsługę w sekcji rekonstrukcji

## 🐛 Troubleshooting

### Problem: Streamlit nie uruchamia się
```bash
pip install streamlit
# lub
pip install --upgrade streamlit
```

### Problem: Nie znajduje modułów src
```python
# Sprawdź czy masz dodane do sys.path w app.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

### Problem: Brak modelu autoencoder.pth
```
💡 Najpierw wytrenuj model używając main.ipynb
💡 Model zostanie zapisany jako autoencoder.pth
```

### Problem: "Module 'streamlit' has no attribute 'cache_resource'"
```bash
# Zaktualizuj Streamlit do wersji >= 1.20
pip install --upgrade streamlit
```

### Problem: Powolne działanie
- Zmniejsz rozmiar obrazu (zmień DEFAULT_IMAGE_SIZE)
- Użyj mniejszego modelu (LightweightSuperRes zamiast SuperResolutionModel)
- Ogranicz liczbę residual blocks w modelu

## 📦 Deployment

### Streamlit Cloud
1. Push kod na GitHub
2. Połącz z Streamlit Cloud
3. Dodaj `requirements.txt`
4. Skonfiguruj secrets (jeśli używasz API keys)

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_gui/app.py"]
```

## 🎨 Dostosowanie wyglądu

### Zmiana motywu
Stwórz `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### Custom CSS
W `app.py`:
```python
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
```

## 📝 Przydatne linki

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io)

---

**Miłej zabawy z GUI! 🚀**
