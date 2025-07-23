# ===================================================
# signal_model.py
# ✅ ResNet1D + SEBlock 기반 ECG 분류
# ✅ labels_df['index'] 기준 split 고정
# ===================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import os

# ✅ SEBlock
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

# ✅ BasicBlock1D
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

# ✅ ResNet1D_SE
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

# ✅ Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return self.alpha * (1 - pt) ** self.gamma * BCE_loss.mean()

# ✅ Dataset
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ Threshold Finder
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(scores)]

# ✅ Evaluate
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            prob = F.softmax(output, dim=1)[:, 1]
            y_true.extend(y_batch.numpy())
            y_prob.extend(prob.numpy())
    best_t = find_best_threshold(np.array(y_true), np.array(y_prob))
    y_pred = (np.array(y_prob) >= best_t).astype(int)
    print(f"✅ Best threshold: {best_t:.2f}")
    print(classification_report(y_true, y_pred))
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1 : {f1_score(y_true, y_pred):.4f}")

# ✅ Main
if __name__ == "__main__":
    set_seed = 42
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)

    label_df = pd.read_excel("./data/labels.xlsx")
    csv_files = sorted(glob("./data/signals/*.csv"), key=lambda x: int(os.path.basename(x).replace(".csv","")))
    label_map = {"Normal": 0, "Abnormal": 1}
    label_dict = label_df.set_index("index")["label"].map(label_map).to_dict()

    X_list, y_list = [], []
    for path in csv_files:
        idx = int(os.path.basename(path).replace(".csv",""))
        signal = pd.read_csv(path, header=None).iloc[:, 0].values
        mean, std = signal.mean(), signal.std()
        signal = (signal - mean) / (std + 1e-8)
        label = label_dict.get(idx)
        if label is not None and not pd.isna(label):
            X_list.append(signal)
            y_list.append(label)

    max_len = max(len(x) for x in X_list)
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype="float32", padding="post")
    X = np.expand_dims(X_padded, axis=1)
    y = np.array(y_list)

    # ✅ 고정 index split (labels_df 기준)
    all_indices = label_df['index'].tolist()
    val_index = [54, 116, 44, 27, 119, 48, 81, 205, 236, 30, 98, 41, 83, 6, 212, 204, 1, 45, 210, 110, 181, 156]
    test_index = [122, 200, 161, 130, 211, 3, 232, 14, 190, 175, 251, 68, 160, 102, 168, 165, 237, 7, 129, 228, 180, 133]
    train_index = list(set(all_indices) - set(val_index) - set(test_index))

    mask_train = label_df['index'].isin(train_index).values
    mask_val   = label_df['index'].isin(val_index).values
    mask_test  = label_df['index'].isin(test_index).values

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_val   = X[mask_val]
    y_val   = y[mask_val]
    X_test  = X[mask_test]
    y_test  = y[mask_test]

    # ✅ train+val 합치기
    X_train_final = np.concatenate([X_train, X_val])
    y_train_final = np.concatenate([y_train, y_val])

    train_loader = DataLoader(ECGDataset(X_train_final, y_train_final), batch_size=8, shuffle=True)
    test_loader  = DataLoader(ECGDataset(X_test, y_test), batch_size=8)

    model = ResNet1D_SE(input_channels=1)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=len(train_loader), epochs=30)

    for epoch in range(30):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"✅ Epoch {epoch+1}/30 done")

    evaluate_model(model, test_loader)
