import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def sobel_edge_detection(image):
    """Apply Sobel edge detection to grayscale image."""
    gray = to_grayscale(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel_combined

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection to grayscale image."""
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def get_transforms():
    """Define image transformations for CNN input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])