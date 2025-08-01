# 🌊 Satellite Image Water Segmentation API (Flask + PyTorch)

A Flask-based web API for segmenting **water bodies** in **12-channel satellite TIF images** using a **U-Net model with a ResNet50 encoder** trained on selected spectral bands.

---

## 🧠 Model Overview

- **Architecture**: U-Net  
- **Encoder**: ResNet50 (ImageNet pre-trained)  
- **Input Channels**: 5 selected bands from 12  
- **Output**: Binary mask (1 = water, 0 = non-water)  
- **Framework**: PyTorch  
- **Loss**: Binary Cross Entropy + Dice (Combo Loss)  

---

## 🧾 Folder Structure

project/
│
├── app.py # Flask web application
├── model.py # U-Net model (ResNet50 encoder)
├── utils.py # Image preprocessing and transformation
├── model.pth # Trained model weights
├── static/ # Static folder to serve output
│ └── example.tif # Example satellite image
├── templates/
│ └── index.html # Upload UI
└── README.md # This file


---
## ⚙️ Requirements
flask
torch
torchvision
opencv-python
tifffile
segmentation-models-pytorch
albumentations
numpy

🛠️ How It Works
Upload a .tif image with 12 channels.

Preprocess: Select 5 bands used during training (e.g. [3, 4, 8, 11, 12]).

Normalize & resize to match model input.

Model Inference with PyTorch U-Net.

Output binary water mask as .png.

✅ Credits
Model architecture: segmentation_models.pytorch

Data: Sentinel-2 or equivalent multi-band satellite images

UI: Flask + HTML template



