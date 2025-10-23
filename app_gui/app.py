"""
GUI w Streamlit do demonstracji działania modeli inpainting i super-resolution.

Użycie:
    streamlit run app_gui/app.py
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

# Dodaj ścieżkę do modułu src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.autoencoder import ConvAutoencoder
from src.models.inpainting_model import UNetInpainting, SimpleInpainting
from src.models.superres_model import SuperResolutionModel, LightweightSuperRes, ESPCNSuperRes
from src.data.damages import (random_mask, rectangular_mask, noise_mask, 
                              line_damage, circular_mask)


# Konfiguracja strony
st.set_page_config(
    page_title="🎨 Art Restoration AI",
    page_icon="🎨",
    layout="wide"
)

# Tytuł
st.title("🎨 Art Restoration AI")
st.markdown("### Uzupełnianie i zwiększanie rozdzielczości dzieł sztuki")
st.markdown("---")


@st.cache_resource
def load_autoencoder(model_path):
    """Ładuje model autoencodera."""
    model = ConvAutoencoder(latent_dim=128)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        return None


@st.cache_resource
def load_inpainting_model(model_path, model_type='unet'):
    """Ładuje model inpainting."""
    if model_type == 'unet':
        model = UNetInpainting()
    else:
        model = SimpleInpainting()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        return None


@st.cache_resource
def load_superres_model(model_path, model_type='standard'):
    """Ładuje model super-resolution."""
    if model_type == 'standard':
        model = SuperResolutionModel(upscale_factor=2)
    elif model_type == 'lightweight':
        model = LightweightSuperRes(upscale_factor=2)
    else:
        model = ESPCNSuperRes(upscale_factor=2)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        return None


def preprocess_image(image, size=(256, 256)):
    """Preprocessuje obraz do formatu tensora."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def apply_damage(img_tensor, damage_type, **kwargs):
    """Aplikuje uszkodzenie do obrazu."""
    img = img_tensor.squeeze(0)
    
    if damage_type == "Losowa plama":
        damaged = random_mask(img, mask_size=kwargs.get('size', 32))
    elif damage_type == "Prostokąt":
        damaged = rectangular_mask(img, mask_ratio=kwargs.get('ratio', 0.2))
    elif damage_type == "Szum":
        damaged = noise_mask(img, noise_level=kwargs.get('level', 0.3))
    elif damage_type == "Linie":
        damaged = line_damage(img, num_lines=kwargs.get('num', 3))
    elif damage_type == "Okrągła plama":
        damaged = circular_mask(img, radius_ratio=kwargs.get('ratio', 0.1))
    else:
        damaged = img
    
    return damaged.unsqueeze(0)


def tensor_to_image(tensor):
    """Konwertuje tensor do obrazu PIL."""
    img = tensor.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# Sidebar - wybór trybu
st.sidebar.header("⚙️ Konfiguracja")

mode = st.sidebar.radio(
    "Wybierz tryb:",
    ["🔧 Inpainting (uzupełnianie)", "🔍 Super-Resolution (zwiększanie rozdzielczości)"]
)

# Upload obrazu
uploaded_file = st.sidebar.file_uploader(
    "Wczytaj obraz dzieła sztuki",
    type=['jpg', 'jpeg', 'png']
)


# ============ TRYB INPAINTING ============
if mode == "🔧 Inpainting (uzupełnianie)":
    st.header("🔧 Inpainting - Uzupełnianie uszkodzeń")
    
    # Wybór typu uszkodzenia
    damage_type = st.sidebar.selectbox(
        "Typ uszkodzenia:",
        ["Losowa plama", "Prostokąt", "Szum", "Linie", "Okrągła plama"]
    )
    
    # Parametry uszkodzenia
    if damage_type == "Losowa plama":
        size = st.sidebar.slider("Rozmiar plamy", 16, 64, 32)
        damage_params = {'size': size}
    elif damage_type == "Prostokąt":
        ratio = st.sidebar.slider("Rozmiar (% obrazu)", 0.1, 0.4, 0.2)
        damage_params = {'ratio': ratio}
    elif damage_type == "Szum":
        level = st.sidebar.slider("Poziom szumu", 0.1, 0.5, 0.3)
        damage_params = {'level': level}
    elif damage_type == "Linie":
        num = st.sidebar.slider("Liczba linii", 1, 10, 3)
        damage_params = {'num': num}
    else:  # Okrągła plama
        ratio = st.sidebar.slider("Promień (% obrazu)", 0.05, 0.2, 0.1)
        damage_params = {'ratio': ratio}
    
    # Wybór modelu
    model_choice = st.sidebar.selectbox(
        "Model:",
        ["Autoencoder (podstawowy)", "U-Net (zaawansowany)", "Simple Inpainting"]
    )
    
    if uploaded_file:
        # Wczytaj obraz
        image = Image.open(uploaded_file).convert('RGB')
        
        # Preprocessuj
        img_tensor = preprocess_image(image)
        
        # Zastosuj uszkodzenie
        damaged_tensor = apply_damage(img_tensor, damage_type, **damage_params)
        
        # Wyświetl obrazy
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Oryginalny")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("💥 Uszkodzony")
            damaged_img = tensor_to_image(damaged_tensor)
            st.image(damaged_img, use_container_width=True)
        
        with col3:
            st.subheader("✨ Odrestaurowany")
            
            # Przycisk do rekonstrukcji
            if st.button("🚀 Rekonstruuj obraz", type="primary"):
                with st.spinner("Przetwarzanie..."):
                    # Załaduj odpowiedni model
                    model_path = "autoencoder.pth"  # Domyślna ścieżka
                    
                    if os.path.exists(model_path):
                        if "Autoencoder" in model_choice:
                            model = load_autoencoder(model_path)
                        elif "U-Net" in model_choice:
                            st.warning("Model U-Net nie jest jeszcze wytrenowany. Używam Autoencodera.")
                            model = load_autoencoder(model_path)
                        else:
                            st.warning("Model Simple Inpainting nie jest jeszcze wytrenowany. Używam Autoencodera.")
                            model = load_autoencoder(model_path)
                        
                        if model:
                            with torch.no_grad():
                                output = model(damaged_tensor)
                                if isinstance(output, tuple):
                                    output = output[0]
                            
                            reconstructed_img = tensor_to_image(output)
                            st.image(reconstructed_img, use_container_width=True)
                            
                            # Metryki (uproszczone)
                            st.success("✅ Rekonstrukcja zakończona!")
                    else:
                        st.error(f"❌ Nie znaleziono modelu: {model_path}")
                        st.info("💡 Wytrenuj model najpierw używając main.ipynb")
    else:
        st.info("👆 Wczytaj obraz używając panelu po lewej")


# ============ TRYB SUPER-RESOLUTION ============
else:
    st.header("🔍 Super-Resolution - Zwiększanie rozdzielczości")
    
    # Wybór stopnia zmniejszenia
    downscale = st.sidebar.slider("Stopień zmniejszenia (do symulacji)", 2, 4, 2)
    
    # Wybór modelu SR
    sr_model_choice = st.sidebar.selectbox(
        "Model Super-Resolution:",
        ["Standard SR", "Lightweight SR", "ESPCN SR"]
    )
    
    if uploaded_file:
        # Wczytaj obraz
        image = Image.open(uploaded_file).convert('RGB')
        
        # Preprocessuj do wysokiej rozdzielczości
        highres_tensor = preprocess_image(image, size=(256, 256))
        
        # Symuluj niską rozdzielczość
        lowres_size = (256 // downscale, 256 // downscale)
        lowres_transform = transforms.Compose([
            transforms.Resize(lowres_size),
            transforms.ToTensor()
        ])
        lowres_tensor = lowres_transform(image).unsqueeze(0)
        
        # Wyświetl obrazy
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Oryginalna jakość")
            st.image(image, use_container_width=True)
            st.caption(f"Rozmiar: 256×256")
        
        with col2:
            st.subheader("📉 Niska rozdzielczość")
            lowres_img = tensor_to_image(lowres_tensor)
            st.image(lowres_img, use_container_width=True)
            st.caption(f"Rozmiar: {lowres_size[0]}×{lowres_size[1]}")
        
        with col3:
            st.subheader("📈 Zwiększona rozdzielczość")
            
            if st.button("🚀 Zwiększ rozdzielczość", type="primary"):
                with st.spinner("Przetwarzanie..."):
                    st.warning("⚠️ Model Super-Resolution nie jest jeszcze wytrenowany.")
                    st.info("💡 Wytrenuj model SR używając odpowiedniego notebooka")
                    
                    # Placeholder - bicubic upsampling
                    upsampled = transforms.Resize((256, 256), 
                                                 interpolation=transforms.InterpolationMode.BICUBIC)
                    bicubic_result = upsampled(lowres_img)
                    st.image(bicubic_result, use_container_width=True)
                    st.caption("(Bicubic interpolation - baseline)")
    else:
        st.info("👆 Wczytaj obraz używając panelu po lewej")


# Stopka
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎨 Art Restoration AI | Projekt uczenia nienadzorowanego</p>
    <p>Modele: Autoencoder + U-Net + Super-Resolution</p>
</div>
""", unsafe_allow_html=True)
