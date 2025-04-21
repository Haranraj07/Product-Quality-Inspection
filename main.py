import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DefectClassifier
from dataset import MVTecDataset
from utils.preprocessing import get_transforms, sobel_edge_detection
from utils.contour_analysis import detect_contours, draw_contours
import os
import cv2
from PIL import Image
import numpy as np

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/trained_model.pth')

def test_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def visualize_defect(image_path, model, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    edge_image = sobel_edge_detection(image)
    bounding_boxes = detect_contours(edge_image)
    output_image = draw_contours(image, bounding_boxes)
    
    # Save visualization
    os.makedirs('outputs/visualizations', exist_ok=True)
    cv2.imwrite('outputs/visualizations/defect_output.png', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

def main():
    # Configuration
    data_dir = 'dataset'
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_dataset = MVTecDataset(data_dir, split='train', transform=get_transforms())
    test_dataset = MVTecDataset(data_dir, split='test', transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    model = DefectClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and test
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    test_model(model, test_loader, device)
    
    # Visualize a sample defective image
    sample_image = 'dataset/test/scratch_head/000.png'
    visualize_defect(sample_image, model, device)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    main()