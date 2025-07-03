import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 1. ECG 시계열 처리용 CNN1D 모듈
class ECGSignalEncoder(nn.Module):
    def __init__(self):
        super(ECGSignalEncoder, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):  # x: [batch, 1, time_steps]
        x = self.cnn1d(x)  # [batch, 128, 1]
        return x.squeeze(-1)  # [batch, 128]

# ✅ 2. ECG 이미지 처리용 CNN2D 모듈
class ECGImageEncoder(nn.Module):
    def __init__(self):
        super(ECGImageEncoder, self).__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):  # x: [batch, 1, H, W]
        x = self.cnn2d(x)  # [batch, 128, 1, 1]
        return x.view(x.size(0), -1)  # [batch, 128]

# ✅ 3. 메타데이터 처리용 MLP 모듈
class MetaDataEncoder(nn.Module):
    def __init__(self, input_dim):
        super(MetaDataEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):  # x: [batch, input_dim]
        return self.mlp(x)  # [batch, 128]

# ✅ 4. Anatomy-Informed Attention Fusion
class AnatomyAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_modalities=3):
        super(AnatomyAttentionFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_modalities))
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 예: normal, abnormal, borderline
        )

    def forward(self, features):  # features: List of [batch, feature_dim]
        # Attention 가중치 normalize (softmax)
        attn = torch.softmax(self.attention_weights, dim=0)  # [num_modalities]
        # 각 modality feature에 attention 곱하기
        fused = torch.cat([f * attn[i] for i, f in enumerate(features)], dim=1)
        return self.fc(fused)

# ✅ 5. 전체 모델
class AnatomyInformedECGNet(nn.Module):
    def __init__(self, meta_input_dim):
        super(AnatomyInformedECGNet, self).__init__()
        self.signal_encoder = ECGSignalEncoder()
        self.image_encoder = ECGImageEncoder()
        self.meta_encoder = MetaDataEncoder(meta_input_dim)
        self.fusion = AnatomyAttentionFusion(feature_dim=128, num_modalities=3)

    def forward(self, ecg_signal, ecg_image, meta_data):
        # 각각의 encoder 통과
        signal_feat = self.signal_encoder(ecg_signal)  # [batch, 128]
        image_feat = self.image_encoder(ecg_image)     # [batch, 128]
        meta_feat = self.meta_encoder(meta_data)       # [batch, 128]
        # Fusion
        out = self.fusion([signal_feat, image_feat, meta_feat])  # [batch, num_classes]
        return out
