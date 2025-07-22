# multimodal.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ✅ Squeeze-and-Excitation (SE) 블록 정의
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# ✅ Basic Residual Block (1D)
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# ✅ ResNet1D with SE
class ResNet1D_SE(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, base_filters=64):
        super(ResNet1D_SE, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = BasicBlock1D(base_filters, base_filters)
        self.layer2 = BasicBlock1D(base_filters, base_filters * 2, stride=2)
        self.layer3 = BasicBlock1D(base_filters * 2, base_filters * 4, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return self.classifier(x)

class ECGMultimodalModel(nn.Module):
    def __init__(self, config):
        super(ECGMultimodalModel, self).__init__()
        self.config = config

        # ✅ Image encoder (ResNet18)
        self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # [B, 512]

        checkpoint = torch.load('./checkpoints/0711_154435/last.pth', map_location='cpu')
        self.image_encoder.load_state_dict(checkpoint, strict=False)

        # ✅ Signal encoder
        # self.signal_encoder = nn.Sequential(
        #     nn.Linear(2490, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 128)
        # )

        self.signal_encoder = ResNet1D_SE(
            input_channels=1,
            num_classes=128,  # 마지막 feature dim을 128로 맞춤
            base_filters=64
        )

        self.load_pretrained_signal_encoder(
            weight_path='./checkpoints/0714_095624/best.pth',
            load_fc=False  # True면 classifier까지, False면 feature extractor만
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
        return 24  # Wt, AGE
    
    def load_pretrained_signal_encoder(self, weight_path, load_fc=False):
        checkpoint = torch.load(weight_path, map_location='cpu')
        if not load_fc:
            checkpoint = {
                k: v for k, v in checkpoint.items()
                # if not k.startswith('classifier')
                if not k.startswith('classifier.4')
            }
        missing, unexpected = self.signal_encoder.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded pretrained signal encoder from {weight_path}")
        if missing:
            print(f"⚠️  Missing keys: {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys: {unexpected}")

    def forward(self, image, ecg_signal, clinical):
        img_feat = self.image_encoder(image)         # [B, 512]
        ecg_signal = ecg_signal.unsqueeze(1)  # [B, L] → [B, 1, L]
        signal_feat = self.signal_encoder(ecg_signal)  # [B, 128]
        clinical_feat = self.clinical_encoder(clinical)  # [B, 32]

        print(f"Image feat: mean={img_feat.mean().item()}, std={img_feat.std().item()}")
        print(f"Signal feat: mean={signal_feat.mean().item()}, std={signal_feat.std().item()}")
        print(f"Clinical feat: mean={clinical_feat.mean().item()}, std={clinical_feat.std().item()}")

        img_logits = self.image_classifier(img_feat)
        signal_logits = self.signal_classifier(signal_feat)
        clinical_logits = self.clinical_classifier(clinical_feat)

        combined = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
        fusion_logits = self.fusion_classifier(combined)

        return img_logits, signal_logits, clinical_logits, fusion_logits
