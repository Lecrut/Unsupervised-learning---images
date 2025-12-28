# Run -> streamlit run gui.py
#%% Imports
import streamlit as st
import torch
import numpy as np
import random
from src.data.damage import DAMAGE_FUNCTIONS
from src.data.load_dataset import load_data
from src.data.cache import load_or_create_damaged_loader, DAMAGED_DATASET_DIR
from src.modules.autoencoder import Autoencoder
from src.modules.replace_damage import replace_damage
from src.modules.pca_clust import our_pca, clustering
from src.modules.inpainter import in_painter_model
from src.modules.super_resolution import SuperResolutionModel
import os
import torch.nn.functional as F

#%% Config
st.set_page_config(layout="wide", page_title="Art Restoration Demo")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Helper Functions
def tensor_to_display_rgb(img_np):
    img_rgb = img_np[:3, :, :]
    img_rgb = np.transpose(img_rgb, (1, 2, 0))
    return np.clip(img_rgb, 0.0, 1.0)
 
#%% Page Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page:", ["Inpainting", "Super Resolution"], index=0)

if page == "Super Resolution":
    try:
        _, test_loader_rgb, _ = load_data(add_fourth_channel=True, num_workers=0)
        sr_model = load_sr_model()
        dataset_rgb = test_loader_rgb.dataset
        
        if 'sr_image_index' not in st.session_state:
            st.session_state['sr_image_index'] = random.randint(0, len(dataset_rgb) - 1)
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()
    
    st.title("Super Resolution Demo")
    
    current_idx = st.session_state['sr_image_index']
    data_item = dataset_rgb[current_idx]
    original_tensor = data_item[0] if isinstance(data_item, (tuple, list)) else data_item
    
    original_tensor = original_tensor.to(device)
    
    if original_tensor.shape[0] == 4:
        img_256 = original_tensor[:3].unsqueeze(0)
    else:
        img_256 = original_tensor.unsqueeze(0)
    
    low_res_128 = F.interpolate(img_256, size=(128, 128), mode='bicubic', align_corners=False)
    
    with torch.no_grad():
        sr_256, _ = sr_model(low_res_128)
        sr_512, _ = sr_model(sr_256)
    
    low_res_128 = low_res_128.squeeze(0).cpu().numpy()
    sr_256 = sr_256.squeeze(0).cpu().numpy()
    sr_512 = sr_512.squeeze(0).cpu().numpy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Low Resolution (128x128)")
        st.image(tensor_to_display_rgb(low_res_128), use_container_width=True, clamp=True)
    
    with col2:
        st.subheader("Super Resolution (256x256)")
        st.image(tensor_to_display_rgb(sr_256), use_container_width=True, clamp=True)
    
    with col3:
        st.subheader("Super Resolution (512x512)")
        st.image(tensor_to_display_rgb(sr_512), use_container_width=True, clamp=True)
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns([1, 3])
    
    with col_btn1:
        if st.button("🎲 Random Image", use_container_width=True, key="sr_random"):
            st.session_state['sr_image_index'] = random.randint(0, len(dataset_rgb) - 1)
            st.rerun()
    
    with col_btn2:
        st.caption(f"Image Index: {st.session_state['sr_image_index']}")
    
    st.stop()

#%% Load Data and Model  (Cached)
@st.cache_resource
def load_resources():   
    with st.spinner('Loading model and data...'):
        _, test_loader_rgb, _ = load_data(add_fourth_channel=False, num_workers=0)
        _, test_loader_rgba, _ = load_data(add_fourth_channel=True, num_workers=0)
        damaged_test_loader = load_or_create_damaged_loader(test_loader_rgba, os.path.join(DAMAGED_DATASET_DIR, 'test'))
        
        autoencoder = Autoencoder(
            latent_dim=768,
            input_channels=4,
            image_size=256,
            learning_rate=0., 
            load_best=True
        )
        autoencoder.eval()
        autoencoder.to(device)
        
    with st.spinner('Extracting latent vectors...'):
        latent_damaged_vectors, _ = autoencoder.extract_latent(damaged_test_loader)
        latent_original_vectors, _ = autoencoder.extract_latent(test_loader_rgba)
        
    with st.spinner('PCA and clustering...'):
        smaller_latent_damaged, smaller_latent_original = our_pca(latent_damaged_vectors, latent_original_vectors)
        clusters_damaged, clusters_original = clustering(smaller_latent_damaged, smaller_latent_original)
        
    return (test_loader_rgb, autoencoder, 
            latent_damaged_vectors, latent_original_vectors,
            smaller_latent_damaged, smaller_latent_original,
            clusters_damaged, clusters_original)

@st.cache_resource
def load_sr_model():
    with st.spinner('Loading Super Resolution model...'):
        sr_model = SuperResolutionModel(
            input_channels=3,
            scale=2,
            learning_rate=0.,
            load_best=True
        )
        sr_model.eval()
        sr_model.to(device)
    return sr_model

def generate_damage_on_the_fly(clean_tensor, seed):
    np.random.seed(seed)
    damage_function = np.random.choice(DAMAGE_FUNCTIONS)
    damaged_tensor = damage_function(clean_tensor)
    return damaged_tensor

def process_image_pipeline(clean_tensor_rgb, model, damage_seed, 
                          latent_original_vectors, clusters_original):
    clean_tensor_rgba = torch.cat([clean_tensor_rgb.to(device), torch.zeros(1, 256, 256).to(device)], dim=0)
    
    damaged_tensor = generate_damage_on_the_fly(clean_tensor_rgba, damage_seed)
    
    damaged_batch = damaged_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        latent_damaged, _ = model.encoder(damaged_batch)
    
    latent_damaged_np = latent_damaged.cpu().numpy()[0]
    
    latent_fixed = in_painter_model(
        [latent_damaged_np],
        [0],
        latent_original_vectors,
        clusters_original
    )
    
    if isinstance(latent_fixed, torch.Tensor):
        latent_fixed = latent_fixed.detach().cpu().numpy()
    
    img_fixed = model.decode_batch(latent_fixed, batch_size=1)[0]
    
    damaged_np = damaged_tensor.cpu().numpy()
    
    fixed_np = replace_damage(damaged_np, img_fixed)
    
    return damaged_np, fixed_np

def main():
    st.title("Inpainting Demo")

    try:
        (test_loader_rgb, model,
         latent_damaged_vectors, latent_original_vectors,
         smaller_latent_damaged, smaller_latent_original,
         clusters_damaged, clusters_original) = load_resources()
        dataset_rgb = test_loader_rgb.dataset
        
        if 'image_index' not in st.session_state:
            st.session_state['image_index'] = random.randint(0, len(dataset_rgb) - 1)
        if 'damage_seed' not in st.session_state:
            st.session_state['damage_seed'] = random.randint(1, 10000)
            
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    current_idx = st.session_state['image_index']
    current_seed = st.session_state['damage_seed']

    data_item = dataset_rgb[current_idx]
    clean_tensor_rgb = data_item[0] if isinstance(data_item, (tuple, list)) else data_item

    damaged_np, fixed_np = process_image_pipeline(clean_tensor_rgb, model, current_seed,
                                                  latent_original_vectors, clusters_original)
    clean_np = clean_tensor_rgb.cpu().numpy()

    with col1:
        st.subheader("Original")
        st.image(tensor_to_display_rgb(clean_np), use_container_width=True)

    with col2:
        st.subheader("Damaged")
        st.image(tensor_to_display_rgb(damaged_np), use_container_width=True)

    with col3:
        st.subheader("Restored")
        st.image(tensor_to_display_rgb(fixed_np), use_container_width=True)

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("🎲 Random Image & Damage", use_container_width=True):
            st.session_state['image_index'] = random.randint(0, len(dataset_rgb) - 1)
            st.session_state['damage_seed'] = random.randint(1, 10000)
            st.rerun()

    with col_btn2:
        if st.button("🔄 Re-roll Damage", use_container_width=True):
            st.session_state['damage_seed'] = random.randint(1, 10000)
            st.rerun()

    with col_btn3:
        st.caption(f"Image Index: {st.session_state['image_index']} | Damage Seed: {st.session_state['damage_seed']}")

if __name__ == "__main__":
    main()