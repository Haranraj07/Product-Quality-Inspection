import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.preprocessing import get_transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load 'good' images
        good_dir = os.path.join(self.root_dir, 'good')
        if os.path.exists(good_dir):
            for img_name in os.listdir(good_dir):
                self.images.append(os.path.join(good_dir, img_name))
                self.labels.append(0)  # Non-defective
        
        # Load defective images
        defect_types = ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']
        for defect in defect_types:
            defect_dir = os.path.join(self.root_dir, defect)
            if os.path.exists(defect_dir):
                for img_name in os.listdir(defect_dir):
                    self.images.append(os.path.join(defect_dir, img_name))
                    self.labels.append(1)  # Defective
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label