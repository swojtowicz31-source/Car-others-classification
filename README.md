# Car vs Other Image Classification with CNN and VGG16

This project implements image classification using Convolutional Neural Networks (CNN) combined with transfer learning based on the VGG16 architecture.  
The goal is to classify images into two categories: **"car"** and **"other"**.

---
## Dataset Structure

Prepare your dataset folder with the following structure:
Each subfolder should contain the respective images for the class.

---
## Setup & Installation

1. Set the `imgdir` variable in the training script (`train_cnn_vgg16.py`) to point to your local dataset path.  
2. Install required Python packages:

```bash
pip install tensorflow numpy matplotlib
python code/train_cnn_vgg16.py
