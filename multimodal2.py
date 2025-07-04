# multimodal.py
import torch
import torch.nn as nn
from torchvision import models

class MultiModalNet(nn.Module):
    def __init__(self, clinical_input_dim, ecg_input_dim, num_classes, dropout=0.3, pretrained=True):
        super(MultiModalNet, self).__init__()
        
        # Image branch
        resnet = models.resnet18(pretrained=pretrained)
        self.image_features = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        self.image_classifier = nn.Linear(resnet.fc.in_features, num_classes)
        
        # Clinical branch
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes)
        )
        
        # ECG branch
        self.ecg_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
        
        # Fusion branch
        self.fusion = nn.Sequential(
            nn.Linear(resnet.fc.in_features + 10 + num_classes, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, img, clinical, ecg):
        # Individual branches
        img_feat = self.image_features(img).view(img.size(0), -1)
        img_output = self.image_classifier(img_feat)
        
        clinical_output = self.clinical_net(clinical)
        
        ecg = ecg.unsqueeze(1)  # add channel dim
        ecg_output = self.ecg_net(ecg)
        
        # Fusion
        fused_features = torch.cat([img_feat, clinical, ecg_output], dim=1)
        fusion_output = self.fusion(fused_features)
        
        return img_output, clinical_output, ecg_output, fusion_output
