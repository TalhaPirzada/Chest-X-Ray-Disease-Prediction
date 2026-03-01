import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Normal Simple Chest X-Ray Analysis",
    page_icon="🩺",
    layout="wide"
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Research", "best_densenet_model.h5")
IMG_SIZE = 320
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the training is complete.")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image):
    # Convert PIL to openCV BGR
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize to model input size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Convert back to 3-channel
    final_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    
    # Samplewise normalization matching training generator
    final_img = final_img.astype('float32')
    mean = np.mean(final_img)
    std = np.std(final_img)
    final_img = (final_img - mean) / (std + 1e-7)
    
    return np.expand_dims(final_img, axis=0)

# Sidebar
st.sidebar.title("🩺 AI Medical Assistant")
st.sidebar.markdown("""
This application uses a **DenseNet121** model trained on the ChestX-ray8 dataset to detect multiple thoracic pathologies.
""")

st.sidebar.subheader("Instructions")
st.sidebar.info("""
1. Upload a Chest X-Ray image (PNG/JPG).
2. The AI will process the image with CLAHE.
3. Prediction scores will be shown for 14 classes.
""")

# Main Content
st.title("Normal Simple Chest X-Ray Analysis Project")
st.write("Upload a radiograph to get AI-assisted screening for various lung conditions.")

uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and show original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model:
        # Preprocess and Predict
        with st.spinner("Analyzing radiograph..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)[0]
            
        with col2:
            st.subheader("Diagnostic Results")
            
            # Sort findings by probability
            findings = sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True)
            
            # Show top findings
            top_threshold = 0.2
            has_positive = any(p >= top_threshold for name, p in findings)
            
            if has_positive:
                st.warning("Pathological findings detected with high probability:")
            else:
                st.success("No strong pathological findings identified.")

            # Create bars for each class
            for name, prob in findings:
                col_name, col_prob = st.columns([2, 5])
                with col_name:
                    st.write(name)
                with col_prob:
                    # Color indicator: intense red for high probability
                    color = "red" if prob > 0.5 else "orange" if prob > 0.2 else "green"
                    st.progress(float(prob))
                    st.caption(f"{prob*100:.1f}% confidence")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational/showcase purposes only. Not for clinical diagnosis.")
