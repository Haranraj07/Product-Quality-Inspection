import torch
import torch.nn as nn
import torchvision.models as models

class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DefectClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)