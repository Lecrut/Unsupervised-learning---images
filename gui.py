# Run -> streamlit run gui.py
#%% Imports
import streamlit as st
import torch
import numpy as np
import random
import os
import torch.nn.functional as F

#%% Importy modułów projektu
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

#%% Resource Loaders
@st.cache_resource
def load_sr_resources():
    with st.spinner('Ładowanie modelu Super Resolution...'):
        _, test_loader, _ = load_data(add_fourth_channel=False, num_workers=0)
        
        sr_model = SuperResolutionModel(
            input_channels=3,
            scale=2,
            learning_rate=0.,
            load_best=True 
        )
        sr_model.eval()
        sr_model.to(device)
    return test_loader, sr_model

@st.cache_resource
def load_ae_resources():
    with st.spinner('Ładowanie modelu Autoencoder i danych...'):
        _, test_loader_rgba, _ = load_data(add_fourth_channel=True, num_workers=0)
        
        autoencoder = Autoencoder(
            input_channels=4,
            load_best=True 
        )
        autoencoder.eval()
        autoencoder.to(device)
        
    return test_loader_rgba, autoencoder

#%% WIDOKI (VIEWS)

def view_home():
    st.title("Centrum Renowacji Sztuki")
    st.markdown("### Witaj w panelu demonstracyjnym")
    st.write("Wybierz odpowiedni moduł z menu po lewej stronie, aby rozpocząć pracę.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎨 **Inpainting (AE)**")
        st.caption("Symulacja uszkodzeń i ekstrakcja cech (Latent Space).")
            
    with col2:
        st.success("🔍 **Super Rozdzielczość**")
        st.caption("Dwukrotne powiększanie obrazu przy użyciu sieci neuronowej.")

    st.write("")
    st.write("")
    st.write("") 
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <h5>Autorzy projektu</h5>
            Filip Lecrut • Piotr Jasiński • Jakub Kocałek
        </div>
        """,
        unsafe_allow_html=True
    )

def view_inpainting():
    st.title("Autoencoder Latent Extraction")
    st.divider()
    
    try:
        test_loader, ae_model = load_ae_resources()
        dataset = test_loader.dataset
    except Exception as e:
        st.error(f"Błąd ładowania Inpaintingu: {e}")
        st.stop()

    if 'inp_image_index' not in st.session_state:
        st.session_state['inp_image_index'] = random.randint(0, len(dataset) - 1)
    if 'inp_damage_seed' not in st.session_state:
        st.session_state['inp_damage_seed'] = random.randint(1, 10000)

    _, col_center, _ = st.columns([3, 2, 3])
    with col_center:
        if st.button("🎲 Losuj nowe zdjęcie", key="btn_inp_new", use_container_width=True):
            st.session_state['inp_image_index'] = random.randint(0, len(dataset) - 1)
            st.session_state['inp_damage_seed'] = random.randint(1, 10000)
            st.rerun()

    current_idx = st.session_state['inp_image_index']
    current_seed = st.session_state['inp_damage_seed']

    data_item = dataset[current_idx]
    original_tensor = data_item[0] if isinstance(data_item, (tuple, list)) else data_item
    
    original_np = tensor_to_display_rgb(original_tensor)

    damaged_tensor = generate_damage_on_the_fly(original_tensor.clone(), current_seed)
    damaged_np = tensor_to_display_rgb(damaged_tensor)

    damaged_batch = damaged_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        latent_vector = ae_model.encoder(damaged_batch)

    fixed_np = np.zeros_like(original_np)
    
    st.write("")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Oryginał")
        st.image(original_np, use_container_width=True)

    with col2:
        st.subheader("2. Uszkodzone")
        st.image(damaged_np, use_container_width=True)
        st.caption(f"Latent shape: {tuple(latent_vector.shape)}")

    with col3:
        st.subheader("3. Naprawione")
        st.image(fixed_np, use_container_width=True)
        st.caption("Moduł naprawczy w budowie")

def view_super_resolution():
    st.title("Super Resolution (x2)")
    st.divider()
    
    try:
        test_loader, sr_model = load_sr_resources()
        dataset = test_loader.dataset
    except Exception as e:
        st.error(f"Błąd ładowania SR: {e}")
        st.stop()
    
    if 'sr_image_index' not in st.session_state:
        st.session_state['sr_image_index'] = random.randint(0, len(dataset) - 1)
    
    _, col_center, _ = st.columns([3, 2, 3])
    with col_center:
        if st.button("🎲 Losuj obraz do SR", key="btn_sr_new", use_container_width=True):
            st.session_state['sr_image_index'] = random.randint(0, len(dataset) - 1)
            st.rerun()

    current_idx = st.session_state['sr_image_index']
    data_item = dataset[current_idx]
    input_tensor = data_item[0] if isinstance(data_item, (tuple, list)) else data_item
    
    if input_tensor.shape[0] == 4:
        input_tensor = input_tensor[:3, :, :]
    
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        sr_output, _ = sr_model(input_batch)
    
    input_np = tensor_to_display_rgb(input_tensor)
    sr_np = tensor_to_display_rgb(sr_output[0])

    st.write("")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wejście (Input)")
        st.image(input_np, caption=f"Rozdzielczość: {input_np.shape[:2]}", use_container_width=False)
    
    with col2:
        st.subheader("Wyjście (Super Res x2)")
        st.image(sr_np, caption=f"Rozdzielczość: {sr_np.shape[:2]}", use_container_width=False)

#%% Main App Logic
def main():
    with st.sidebar:
        st.title("Nawigacja")
        
        selected_page = st.radio(
            "Wybór widoku", 
            ["Strona Główna", "Odtwarzanie", "Super Rozdzielczość"],
            label_visibility="collapsed"
        )
        
        st.markdown(
            """
            <style>
                .sidebar-footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    width: 20rem;
                    padding: 20px;
                    text-align: center;
                    color: grey;
                    font-size: 12px;
                    background-color: transparent;
                    pointer-events: none;
                    z-index: 100;
                }
            </style>
            <div class="sidebar-footer">
                Art Restoration Project v1.0
            </div>
            """,
            unsafe_allow_html=True
        )

    if selected_page == "Strona Główna":
        view_home()
    elif selected_page == "Odtwarzanie":
        view_inpainting()
    elif selected_page == "Super Rozdzielczość":
        view_super_resolution()

if __name__ == "__main__":
    main()