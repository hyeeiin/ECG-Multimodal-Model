# multimodal.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ECGMultimodalModel(nn.Module):
    def __init__(self, config):
        super(ECGMultimodalModel, self).__init__()
        self.config = config

        # ✅ Image encoder (ResNet18)
        self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # [B, 512]

        # ✅ Signal encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(2490, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )

        # ✅ Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(self.get_clinical_feature_dim(), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ✅ Branch classifiers
        self.image_classifier = nn.Linear(512, config.num_classes)
        self.signal_classifier = nn.Linear(128, config.num_classes)
        self.clinical_classifier = nn.Linear(32, config.num_classes)

        # ✅ Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes)
        )

    def get_clinical_feature_dim(self):
        return 2  # Wt, AGE

    def forward(self, image, ecg_signal, clinical):
        img_feat = self.image_encoder(image)         # [B, 512]
        signal_feat = self.signal_encoder(ecg_signal)  # [B, 128]
        clinical_feat = self.clinical_encoder(clinical)  # [B, 32]

        img_logits = self.image_classifier(img_feat)
        signal_logits = self.signal_classifier(signal_feat)
        clinical_logits = self.clinical_classifier(clinical_feat)

        combined = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
        fusion_logits = self.fusion_classifier(combined)

        return img_logits, signal_logits, clinical_logits, fusion_logits
