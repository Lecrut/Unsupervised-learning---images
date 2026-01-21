# Run: streamlit run gui.py
#%% Imports
import streamlit as st
import torch
import numpy as np
import random
import os

# --- TWOJE ORYGINALNE IMPORTY ---
from src.data.damage import DAMAGE_FUNCTIONS
from src.data.load_dataset import load_data
from src.modules.autoencoder import Autoencoder
from src.modules.super_resolution import SuperResolutionModel

#%% Config
st.set_page_config(
    layout="wide", 
    page_title="Art Restoration Demo",
    initial_sidebar_state="expanded"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- POPRAWIONY CSS ---
st.markdown("""
    <style>
        /* Import czcionek */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
        
        /* 1. NAPRAWA PRZYCISKU SIDEBARA (ZACHOWANIE FUNKCJONALNOŚCI) */
        [data-testid="stSidebarCollapsedControl"] {
            color: transparent !important;
            background: transparent !important;
            border: none !important;
            width: auto !important;
            min-width: 80px !important;
        }

        /* Ukrywamy oryginalną ikonę, bo ona znikała przez zmianę fontu */
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] img {
            display: none !important;
        }

        /* Dodajemy własny tekst "MENU" jako przycisk */
        [data-testid="stSidebarCollapsedControl"]::after {
            content: "MENU";
            color: #0F2C59 !important; 
            font-family: 'Montserrat', sans-serif !important;
            font-size: 13px !important;
            font-weight: 600;
            letter-spacing: 0.5px;
            background: rgba(15, 44, 89, 0.08);
            border-radius: 6px;
            padding: 8px 16px;
            display: inline-block;
            white-space: nowrap;
            cursor: pointer;
        }

        [data-testid="stSidebarCollapsedControl"]:hover::after {
            background: rgba(15, 44, 89, 0.15);
        }

        /* 2. PRECYZYJNE PRZYPISANIE CZCIONEK (BY NIE PSUĆ IKON) */
        
        .stApp { background-color: #F8F9FA; }
        section[data-testid="stSidebar"] { background-color: #EBF1F7; border-right: 1px solid #D1D9E6; }
        
        /* Nagłówki */
        h1, h2, h3 { 
            font-family: 'Playfair Display', serif !important; 
            color: #0F2C59 !important; 
        }

        /* WAŻNE: Przypisujemy font tylko do kontenerów tekstowych, 
           a nie do globalnych tagów 'span' czy 'div', co psuło ikony. */
        
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li, 
        [data-testid="stMarkdownContainer"] div,
        [data-testid="stCaptionContainer"] {
            font-family: 'Montserrat', sans-serif !important;
        }

        /* Twoje klasy niestandardowe */
        .authors-box, .footer-text, .custom-text {
            font-family: 'Montserrat', sans-serif !important;
        }

        /* Nawigacja (Radio) */
        div[role="radiogroup"] > label > div:first-child { display: none !important; }
        div[role="radiogroup"] label {
            font-family: 'Montserrat', sans-serif !important;
            background: transparent !important; 
            border: none !important;
            padding: 8px 16px; 
            margin-bottom: 2px;
            color: #0F2C59 !important; 
            font-size: 16px;
            transition: transform 0.2s;
        }
        div[role="radiogroup"] label:hover { transform: translateX(5px); font-weight: 600 !important; }
        div[role="radiogroup"] label[data-checked="true"] { font-weight: 700 !important; }
        div[role="radiogroup"] label p { color: #0F2C59 !important; font-family: 'Montserrat', sans-serif !important; }

        /* Przyciski */
        .stButton > button {
            font-family: 'Montserrat', sans-serif !important;
            background: #0F2C59 !important; 
            color: white !important;
            border-radius: 8px; 
            border: none; 
            padding: 10px 20px;
        }
        .stButton > button:hover { background: #1B3C73 !important; }

        /* Styl kart autorskich */
        .authors-box {
            background: white; padding: 20px; border-radius: 12px;
            border-left: 5px solid #0F2C59; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center; margin-top: 20px;
        }
        .footer-text { text-align: center; color: #64748B; font-size: 11px; margin-top: 10px; opacity: 0.8; }
        
        header[data-testid="stHeader"] { background: transparent !important; }
    </style>
""", unsafe_allow_html=True)


#%% Helper Functions
def tensor_to_display_rgb(img_np):
    if isinstance(img_np, torch.Tensor):
        img_np = img_np.detach().cpu().numpy()
    if img_np.shape[0] == 4: 
        img_np = img_np[:3, :, :]
    img_rgb = np.transpose(img_np, (1, 2, 0))
    return np.clip(img_rgb, 0.0, 1.0)

def generate_damage_on_the_fly(clean_tensor, seed):
    np.random.seed(seed)
    damage_function = np.random.choice(DAMAGE_FUNCTIONS)
    damaged_tensor = damage_function(clean_tensor)
    return damaged_tensor

#%% Loaders
@st.cache_resource
def load_sr_resources():
    with st.spinner('Ładowanie modelu Super Resolution...'):
        _, test_loader, _ = load_data(add_fourth_channel=False, num_workers=0)
        # UWAGA: Upewnij się, że ścieżki i parametry w SuperResolutionModel są poprawne dla Twojego projektu
        sr_model = SuperResolutionModel(input_channels=3, scale=2, learning_rate=0., load_best=True)
        sr_model.eval()
        sr_model.to(device)
    return test_loader, sr_model

@st.cache_resource
def load_ae_resources():
    with st.spinner('Ładowanie modelu Autoencoder...'):
        _, test_loader_rgba, _ = load_data(add_fourth_channel=True, num_workers=0)
        # UWAGA: Upewnij się, że parametry Autoencoder są poprawne
        autoencoder = Autoencoder(input_channels=4, load_best=True)
        autoencoder.eval()
        autoencoder.to(device)
    return test_loader_rgba, autoencoder

#%% VIEWS

def view_home():
    st.markdown("<h1>Centrum Renowacji Sztuki</h1>", unsafe_allow_html=True)
    st.write("Wybierz narzędzie z menu po lewej stronie.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎨 **Inpainting**\n\nSymulacja uszkodzeń i ekstrakcja cech.")
    with col2:
        st.success("🔍 **Super Rozdzielczość**\n\nUpscaling obrazów (x2).")

    st.markdown("""
        <div class="authors-box">
            <div style="font-weight: bold; color: #0F2C59; margin-bottom: 5px;">Zespół Projektowy</div>
            Filip Lecrut • Piotr Jasiński • Jakub Kocałek
        </div>
    """, unsafe_allow_html=True)

def view_inpainting():
    st.markdown("<h1>Autoencoder Latent Extraction</h1>", unsafe_allow_html=True)
    st.divider()
    
    try:
        test_loader, ae_model = load_ae_resources()
        dataset = test_loader.dataset
    except Exception as e:
        st.error(f"Błąd ładowania zasobów Autoencodera: {e}")
        st.stop()

    if 'inp_idx' not in st.session_state: st.session_state['inp_idx'] = random.randint(0, len(dataset) - 1)
    if 'inp_seed' not in st.session_state: st.session_state['inp_seed'] = random.randint(1, 10000)

    _, c2, _ = st.columns([3, 2, 3])
    if c2.button("Losuj obraz", key="btn_inp"):
        st.session_state['inp_idx'] = random.randint(0, len(dataset) - 1)
        st.session_state['inp_seed'] = random.randint(1, 10000)
        st.rerun()

    data_item = dataset[st.session_state['inp_idx']]
    orig_t = data_item[0] if isinstance(data_item, (tuple, list)) else data_item
    dmg_t = generate_damage_on_the_fly(orig_t.clone(), st.session_state['inp_seed'])
    
    with torch.no_grad():
        latent = ae_model.encoder(dmg_t.unsqueeze(0).to(device))

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1. Oryginał")
    c1.image(tensor_to_display_rgb(orig_t), use_container_width=True)
    
    c2.markdown("### 2. Uszkodzone")
    c2.image(tensor_to_display_rgb(dmg_t), use_container_width=True)
    c2.caption(f"Latent: {tuple(latent.shape)}")
    
    c3.markdown("### 3. Naprawione")
    c3.image(np.zeros_like(tensor_to_display_rgb(orig_t)), use_container_width=True)
    c3.caption("Wkrótce...")

def view_sr():
    st.markdown("<h1>Super Resolution (x2)</h1>", unsafe_allow_html=True)
    st.divider()
    
    try:
        test_loader, sr_model = load_sr_resources()
        dataset = test_loader.dataset
    except Exception as e:
        st.error(f"Błąd ładowania zasobów SR: {e}")
        st.stop()

    if 'sr_idx' not in st.session_state: st.session_state['sr_idx'] = random.randint(0, len(dataset) - 1)

    _, c2, _ = st.columns([3, 2, 3])
    if c2.button("Losuj obraz", key="btn_sr"):
        st.session_state['sr_idx'] = random.randint(0, len(dataset) - 1)
        st.rerun()

    data_item = dataset[st.session_state['sr_idx']]
    input_t = data_item[0] if isinstance(data_item, (tuple, list)) else data_item
    if input_t.shape[0] == 4: input_t = input_t[:3]
    
    with torch.no_grad():
        out = sr_model(input_t.unsqueeze(0).to(device))

    st.write("")
    c1, c2 = st.columns(2)
    c1.markdown("### Wejście")
    c1.image(tensor_to_display_rgb(input_t), caption=f"Rozdzielczość: {input_t.shape[1:]}")
    c2.markdown("### Wyjście (x2)")
    c2.image(tensor_to_display_rgb(out[0]), caption=f"Rozdzielczość: {out[0].shape[1:]}")

#%% Main
def main():
    with st.sidebar:
        # Tytuł nawigacji
        st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Nawigacja</h1>", unsafe_allow_html=True)
        
        # Menu (Zwykły tekst, stylizowany CSSem)
        page = st.radio("Nav", ["Strona Główna", "Odtwarzanie", "Super Rozdzielczość"], label_visibility="collapsed")
        
        st.write("")
        st.write("")
        st.divider()
        st.markdown("<div class='footer-text'>ART RESTORATION PROJECT v1.0</div>", unsafe_allow_html=True)

    if page == "Strona Główna": view_home()
    elif page == "Odtwarzanie": view_inpainting()
    elif page == "Super Rozdzielczość": view_sr()

if __name__ == "__main__":
    main()