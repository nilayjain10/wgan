import streamlit as st
import torch
import numpy as np
from model import Generator

# --- Configuration ---
st.set_page_config(page_title="WGAN-GP CIFAR-10", page_icon="🎨")
st.title("🎨 WGAN-GP Image Generator")
st.markdown("This app uses a **Wasserstein GAN with Gradient Penalty** to generate synthetic images based on the CIFAR-10 dataset.")

# --- Load Model ---
@st.cache_resource
def load_model():
    # Ensure these match your notebook hyperparameters
    Z_DIM = 100
    FEATURES_GEN = 64
    IMG_CHANNELS = 3
    
    model = Generator(Z_DIM, FEATURES_GEN, IMG_CHANNELS)
    # Load to CPU for local use
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Sidebar ---
st.sidebar.header("Generation Settings")
num_images = st.sidebar.slider("Number of images to generate", 1, 16, 4)
grid_cols = st.sidebar.columns(1)[0]

# --- Main Logic ---
if st.button("✨ Generate Images"):
    try:
        gen = load_model()
        
        # Generate random latent vectors
        noise = torch.randn(num_images, 100, 1, 1)
        
        with torch.no_grad():
            fake_tensors = gen(noise)
        
        # Display images in a responsive grid
        cols = st.columns(4) 
        for i in range(num_images):
            # Denormalize from [-1, 1] to [0, 1]
            img = fake_tensors[i].permute(1, 2, 0).numpy()
            img = (img * 0.5) + 0.5
            img = np.clip(img, 0, 1) # Ensure valid pixel range
            
            with cols[i % 4]:
                st.image(img, use_container_width=True, caption=f"Generated {i+1}")
                
    except FileNotFoundError:
        st.error("Error: 'generator.pth' not found. Please download it from Kaggle and place it in this folder.")