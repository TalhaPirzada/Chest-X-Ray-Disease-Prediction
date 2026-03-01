import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Research", "best_densenet_model.h5")
IMG_SIZE = 320
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Load model globally to avoid reloading
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

def preprocess_image(image):
    # Gradio provides PIL Image or numpy array depending on source
    img = np.array(image)
    
    # Ensure 3-channel BGR for consistency with CV2 processing
    if len(img.shape) == 2: # Gray
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else: # RGB
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

def predict_diagnosis(img):
    if model is None:
        return "Model not found. Please check Research/best_densenet_model.h5"
    
    if img is None:
        return None

    # Preprocess
    processed_img = preprocess_image(img)
    
    # Predict
    predictions = model.predict(processed_img)[0]
    
    # Map to dictionary for Gradio Label output
    results = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    return results

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Normal Simple Chest X-Ray Analysis Project")
    gr.Markdown("### A clean, functional prototype to detect thoracic pathologies from X-ray radiographs.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload X-Ray Radiograph")
            predict_btn = gr.Button("Analyze Image", variant="primary")
            gr.Examples(
                examples=[], # You can add paths to sample images here
                inputs=input_img
            )
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Top Pathological Findings")
            
    gr.Markdown("---")
    gr.Markdown("#### Pathologies detected:")
    gr.Markdown(", ".join(CLASS_NAMES))
    gr.Markdown("*Note: This application is for demonstration purposes and should not be used for actual medical diagnosis.*")

    predict_btn.click(
        fn=predict_diagnosis,
        inputs=input_img,
        outputs=output_label
    )

if __name__ == "__main__":
    demo.launch(share=True)
