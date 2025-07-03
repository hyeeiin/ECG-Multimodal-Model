import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

# ─── 데이터셋 ───
class MultiECGDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.signals = torch.tensor(df.iloc[:, 1:1+L].values, dtype=torch.float32).unsqueeze(1)
        self.img_paths = df['image_name'].values
        self.meta = torch.tensor(df[['age','sex','...']].values, dtype=torch.float32)
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = Image.open(f"{self.img_dir}/{self.img_paths[idx]}").convert('L')
        if self.transform: img = self.transform(img)
        return self.signals[idx], img, self.meta[idx], self.labels[idx]

# ─── 모델 ───
class MultiModalFCA(nn.Module):
    def __init__(self, signal_len, meta_dim, num_classes=3):
        super().__init__()
        # CNN1D for ECG signal
        self.cnn1d = nn.Sequential(
            nn.Conv1d(1,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        # CNN2D for image
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Metadata MLP
        self.mlp_meta = nn.Sequential(nn.Linear(meta_dim,32), nn.ReLU())
        # Fusion & FCA block
        fused_dim = 64 + 32 + 32
        self.fca = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                  nn.Sigmoid())
        # Classification
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, sig, img, meta):
        s = self.cnn1d(sig).mean(dim=2)
        i = self.cnn2d(img).squeeze(-1).squeeze(-1)
        m = self.mlp_meta(meta)
        concat = torch.cat([s,i,m], dim=1)
        weights = self.fca(concat)
        fused = concat * weights
        return self.classifier(fused)

# ─── 학습 루프 ───
# dataset, loader 설정 후
model = MultiModalFCA(signal_len=L, meta_dim=M)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for sig, img, meta, label in loader:
        out = model(sig, img, meta)
        loss = criterion(out, label)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
