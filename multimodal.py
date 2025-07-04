# multimodal.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ECGMultimodalModel(nn.Module):
    def __init__(self, config):
        super(ECGMultimodalModel, self).__init__()
        self.config = config

        # ✅ Image branch: pretrained ResNet
        self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # remove final fc → feature output shape: [B, 512]

        # ✅ Signal branch (예: 시계열 길이: 2490)
        self.signal_encoder = nn.Sequential(
            nn.Linear(2490, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )

        # ✅ Clinical branch
        # clinical feature 수는 dataset.py에서 .shape[1] 출력해서 맞추기!
        self.clinical_encoder = nn.Sequential(
            nn.Linear(self.get_clinical_feature_dim(), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ✅ Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes)
        )

    def get_clinical_feature_dim(self):
        # dataset.py에서 실제 clinical scaler.shape[1]과 맞춰야 안전!
        return 19  # 예시: 실제 컬럼 수로 수정

    def forward(self, image, ecg_signal, clinical):
        img_feat = self.image_encoder(image)
        signal_feat = self.signal_encoder(ecg_signal)
        clinical_feat = self.clinical_encoder(clinical)
        combined = torch.cat((img_feat, signal_feat, clinical_feat), dim=1)
        return self.classifier(combined)
