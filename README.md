# gpu-capstone-change-detection
# 🚀 GPU-Based Image Segmentation using U-Net (PyTorch + Colab + CUDA)

## ✅ Project Overview
This project demonstrates how to train a U-Net deep learning model for image segmentation using **GPU acceleration**.  
The model is trained using **PyTorch with CUDA** on Google Colab.  
The goal is to show understanding of GPU-based training and deployment of AI models, as required in the **Coursera GPU Specialization Capstone Project**.

✅ Uses **GPU-enabled training (CUDA)**  
✅ Runs in **Google Colab** or any machine with NVIDIA GPU  
✅ Works even with **unlabeled images** (auto-generated masks)  
✅ Includes **training script, inference script, model saving, plots, sample outputs**

---

## 📂 Repository Structure

```
gpu-capstone-change-detection/
│
├── data/                # Input images (img1.jpg, img2.jpg, etc.)
├── models/
│   └── unet.py          # U-Net model architecture
├── artifacts/           # Output folder: trained model + loss graph + predictions
├── train.py             # Training script (GPU enabled)
├── inference.py         # Runs model inference + saves predicted mask
├── requirements.txt     # Required Python packages
└── README.md            # Project description
```
---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
Libraries included:

-torch

-torchvision

-numpy

-matplotlib

-opencv-python

-tqdm

-albumentations

-Pillow

---

## 🖥️ How to Run This Project in Google Colab

### ✅ 1. Enable GPU  
In Colab:  
`Runtime > Change runtime type > Hardware accelerator > GPU`

### ✅ 2. Clone this repository

```python
!git clone https://github.com/Jisha-ji/gpu-capstone-change-detection.git
%cd gpu-capstone-change-detection
```

### ✅ 3. Install dependencies
```!pip install -r requirements.txt```

### ✅ 4. Run Training (saves model + loss plot in artifacts/)
```!python train.py```


After training, files will be saved in:

artifacts/
 ├── unet_simple.pth
 └── training_loss.png

### 🔍 Run Inference (Test the Trained Model)

```!python inference.py```


Inference output saved as:

artifacts/output_mask.png

### 🧠 Model Used: U-Net

U-Net is an encoder–decoder convolutional neural network commonly used for image segmentation tasks like medical imaging and satellite analysis.
It works by compressing the input image into features and then expanding it back to predict a pixel-level mask.

### 🎯 Coursera Submission Checklist

✅ GitHub repository contains all required files

✅ Training successfully executed on GPU (proof via nvidia-smi)

✅ Model artifacts created and committed

✅ README includes all documentation

✅ 5–10 minute demo video prepared

### 👤 Author

Name: Jisha
Course: CUDA at Scale for the Enterprise – Capstone Project
GitHub: https://github.com/Jisha-ji

🔁 Feel free to fork this repo or create a pull request if you'd like to contribute!


---

### ✅ Final Step

1. Replace your current README.md with this version.
2. Click **Commit changes** in GitHub.

