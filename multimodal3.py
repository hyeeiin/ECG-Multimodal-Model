import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ ECG 시계열 처리용 CNN1D 모듈
class ECGSignalEncoder(nn.Module):
    def __init__(self):
        super(ECGSignalEncoder, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
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

# ✅ ECG 이미지 처리용 CNN2D 모듈
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

# ✅ 메타데이터 처리용 MLP 모듈
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

# ✅ Graph Attention Layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)  # [batch, num_nodes, out_features]
        N = Wh.size(1)

        a_input = torch.cat([Wh.repeat(1, 1, N).view(-1, N * N, Wh.size(2)), 
                             Wh.repeat(1, N, 1)], dim=2)
        e = self.leakyrelu(self.a(a_input).squeeze(2))  # [batch, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # [batch, N, out_features]
        return h_prime

# ✅ 전체 모델
class AnatomyInformedGATNet(nn.Module):
    def __init__(self, meta_input_dim, num_classes=3):
        super(AnatomyInformedGATNet, self).__init__()
        self.signal_encoder = ECGSignalEncoder()
        self.image_encoder = ECGImageEncoder()
        self.meta_encoder = MetaDataEncoder(meta_input_dim)

        self.gat = GraphAttentionLayer(in_features=128, out_features=128)

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, ecg_signal, ecg_image, meta_data, adj):
        # Encode each modality
        signal_feat = self.signal_encoder(ecg_signal).unsqueeze(1)  # [batch, 1, 128]
        image_feat = self.image_encoder(ecg_image).unsqueeze(1)     # [batch, 1, 128]
        meta_feat = self.meta_encoder(meta_data).unsqueeze(1)       # [batch, 1, 128]

        # Concatenate as graph nodes
        nodes = torch.cat([signal_feat, image_feat, meta_feat], dim=1)  # [batch, 3, 128]

        # Apply Graph Attention
        fused_features = self.gat(nodes, adj).view(nodes.size(0), -1)  # [batch, 3*128]

        # Final classification
        out = self.fc(fused_features)
        return out
