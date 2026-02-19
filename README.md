# 🩻 Medical AI: Pneumonia Diagnosis with Grad-CAM Interpretability

## 📖 Overview
This project is an AI-assisted diagnostic tool that classifies Chest X-ray images into three categories: Normal, Viral Pneumonia, and Bacterial Pneumonia. 

**The Highlight:** To bridge the trust gap in medical AI, this tool implements **Grad-CAM**, providing heatmaps that visualize exactly where the model is looking to make its diagnosis.

## 🛠️ Tech Stack
- **Core**: PyTorch, Torchvision
- **Model Architecture**: ResNet50 (Transfer Learning)
- **Interpretability**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Frontend**: Streamlit
- **Acceleration**: Apple Silicon MPS / NVIDIA CUDA

## 🧠 Model Pipeline
- **Training**: Fine-tuned ResNet50 on a multi-class pneumonia dataset.
- **Data Augmentation**: Applied random rotations, flips, and normalization to improve generalization.
- **Explainability**: Custom Grad-CAM hook to extract feature maps from the final convolutional layer.

## 🚀 How to Run
- **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/medical-ai.git](https://github.com/YOUR_USERNAME/medical-ai.git)
   cd medical-ai
