# DaVinci Face: Unified Multi-Task Face Analysis via Spatial and Channel Cross-Attention


Official PyTorch implementation of **DaVinci Face**, an efficient, unified framework for real-time multi-task face analysis. This repository provides the codebase for the model architecture, multi-task training loop, and inference scripts, alongside the model weights.

## Overview
DaVinci Face is a multi-task face analysis framework that utilizes both spatial cross-attention and channel cross-attention to extract local and global features.

Operating at **270 FPS** with only **42.34M parameters** and **8.46 GFLOPs**, it simultaneously performs 9 tasks:
* Face Verification
* Facial Expression Recognition
* Age Estimation
* Gender Recognition
* Race Recognition
* Attribute Estimation
* Head Pose Estimation
* Landmark Detection
* Visibility Detection

## Installation and Inference

1. Clone the repository:
```bash
git clone https://github.com/Thoalfeqar-gata/DaVinci-Face
cd DaVinci-Face
```
2. Install PyTorch with CUDA support (adjust the CUDA version to match your hardware):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

3. Install the remaining dependencies
```bash
   pip install -r requirements.txt
```

4. Download the model from:
    https://drive.google.com/drive/folders/1UW7vPLoMooipEjuYskYEb8JNgKBLAzC3?usp=drive_link

5. Place the model.pth file in the data/models/DaVinci Face directory.

6. To run the model, open inference_DaVinciFace.py and run it. The main function has an image_path variable that you can set to the path of the image you want to use for inference.

## Training 

1. To train the model, check the data preparation folder, which includes notebooks that provide links for the datasets used and the preprocessing pipeline used for each dataset.
2. After preparing the data, run the train_DaVinciFace.py file.