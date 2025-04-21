# Product Quality Inspection using Deep CNN and Edge Detection

This project implements an automated defect detection system for manufactured products using Convolutional Neural Networks (CNNs) and edge detection techniques. It leverages the MVTec AD dataset for training and testing.

## Folder Structure
- `dataset/`: MVTec AD dataset (e.g., screw category).
- `models/`: Trained CNN models.
- `utils/`: Preprocessing and contour analysis scripts.
- `main.py`: Main script for training, testing, and visualization.
- `model.py`: CNN model architecture.
- `dataset.py`: Data loader for MVTec AD.
- `gradcam.py`: Grad-CAM visualization (optional).
- `requirements.txt`: Python dependencies.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt