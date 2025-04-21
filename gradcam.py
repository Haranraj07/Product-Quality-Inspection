import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(grad):
            self.gradients = grad
        
        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(save_gradient)
        
        target_layer.register_forward_hook(forward_hook)
    
    def generate(self, input_image, class_idx=None):
        self.model.eval()
        input_image = input_image.unsqueeze(0)
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        output[:, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        return heatmap

def visualize_gradcam(image_path, model, target_layer, output_path='outputs/visualizations/gradcam.png'):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(input_image)
    
    # Overlay heatmap on original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image
    cv2.imwrite(output_path, superimposed_img)