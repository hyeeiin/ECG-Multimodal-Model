# multimodal.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ECGMultimodalModel(nn.Module):
    def __init__(self, config):
        super(ECGMultimodalModel, self).__init__()
        self.config = config

        # ✅ 최신 버전: weights 인자로 명시적으로 전달
        self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # Remove final FC for feature extraction

        self.signal_encoder = nn.Sequential(
            nn.Linear(config.num_ecg_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(config.num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes)
        )

    def forward(self, image, ecg_signal, clinical):
        img_feat = self.image_encoder(image)
        signal_feat = self.signal_encoder(ecg_signal)
        clinical_feat = self.clinical_encoder(clinical)

        combined = torch.cat((img_feat, signal_feat, clinical_feat), dim=1)
        out = self.classifier(combined)
        return out
