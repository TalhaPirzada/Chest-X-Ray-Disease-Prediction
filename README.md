 # 🩺 Normal Simple Chest X-Ray Analysis Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-App-orange.svg)](https://gradio.app/)

A deep learning-based medical imaging project designed to assist radiologists in detecting and screening for thoracic pathologies from Chest X-Ray radiographs. Built using **DenseNet121** and trained on the **NIH ChestX-ray8 dataset**.

This project focuses on a "normal simple" implementation, prioritizing a clean image processing pipeline (CLAHE) and a functional UI over complex model tuning.

---

## 🚀 Key Features

*   **Multi-Label Classification**: Detects up to 14 conditions simultaneously in a single scan.
*   **Deep Learning Backbone**: Leverages the power of **DenseNet121** for high-precision feature extraction.
*   **Smart Preprocessing**: Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast, making subtle pathological features more visible.
*   **Dual Interface**: 
    *   **Streamlit Dashboard**: A professional, feature-rich web interface for diagnostic analysis.
    *   **Gradio Interface**: A quick, user-friendly UI for rapid testing and sharing via public links.
*   **Research Notebooks**: Includes comprehensive data exploration, leakage analysis (patient overlap), and training workflows.

---

## 🦠 Pathologies Detected

The AI model is trained to identify the following conditions:

| Pathologies | | | |
| :--- | :--- | :--- | :--- |
| 🫁 Atelectasis | 🫀 Cardiomegaly | ☁️ Consolidation | 💧 Edema |
| 🌊 Effusion | 🌬️ Emphysema | 🧶 Fibrosis | 🩺 Hernia |
| 🌫️ Infiltration | 🌑 Mass | 🔘 Nodule | 🛡️ Pleural Thickening |
| 🦠 Pneumonia | 🎈 Pneumothorax | | |

---

## 🛠️ Tech Stack

*   **Core**: Python
*   **Deep Learning**: TensorFlow, Keras
*   **Image Processing**: OpenCV, PIL
*   **Data Science**: Pandas, NumPy, Scikit-learn
*   **Visualization**: Matplotlib, Seaborn
*   **Deployment**: Streamlit, Gradio

---

## 📂 Project Structure

```text
├── Research/
│   ├── best_densenet_model.h5   # Pre-trained core model
│   └── testv1.ipynb             # Research, training, and evaluation notebook
├── data/
│   └── nih/                     # Dataset metadata and samples
├── app.py                       # Streamlit Web Application
├── gradio_app.py                # Gradio Web Interface
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Launch Streamlit App (Recommended)
This provides a full dashboard with confidence meters and diagnostic summaries.
```bash
streamlit run app.py
```

### 2. Launch Gradio Interface
Best for quick testing and generating a shareable public URL.
```bash
python gradio_app.py
```

### 3. Explore Research
To view the training process, data exploration, and performance metrics, open:
`Research/testv1.ipynb`

---

## 🔬 Model Information

*   **Architecture**: DenseNet121 (pre-trained on ImageNet, fine-tuned).
*   **Input Size**: 320x320 pixels.
*   **Dataset**: NIH ChestX-ray8 (112,120 X-ray images from 30,805 unique patients).
*   **Performance Optimization**: Implemented binary cross-entropy with weighted loss to handle class imbalance.

---

## ⚠️ Disclaimer

*This tool is for **educational and research purposes only**. It is not intended for clinical use or to replace the professional judgment of a qualified radiologist.*

---
Designed with ❤️ for AI in Healthcare.
