import cv2
import numpy as np

def detect_contours(edge_image):
    """Detect contours in edge-detected image and return bounding boxes."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def draw_contours(image, bounding_boxes):
    """Draw bounding boxes on the original image."""
    image_copy = np.array(image)
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_copy   