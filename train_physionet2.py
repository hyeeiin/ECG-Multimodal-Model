# ===================================================
# ETH Zürich CRNN 버전 train_signal_only.py
# ===================================================

import os
import time
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import stft

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve

import matplotlib.pyplot as plt

from config import Config


# ===================================================
# ✅ 1️⃣ Log-Spectrogram 함수
# ===================================================

def compute_log_spectrogram(signal, fs=300, window='tukey', nperseg=64, noverlap=32):
    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.abs(Zxx)
    log_Sxx = np.log1p(Sxx)  # log(1 + |X|)
    return log_Sxx  # shape: (freq_bins, time_bins)


# ===================================================
# ✅ 2️⃣ Dataset 클래스
# ===================================================

class ECGDatasetSpectrogram(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================================================
# ✅ 3️⃣ CRNN 모델 (ConvBlock + BiLSTM)
# ===================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5,5), padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
    def forward(self, x):
        return self.block(x)

class CRNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CRNN, self).__init__()
        self.conv1 = ConvBlock(input_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        self.flatten = nn.Flatten(start_dim=2)  # keep batch & channel
        self.bilstm = nn.LSTM(input_size=512, hidden_size=200, num_layers=3,
                              batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(400, 64),  # 200 × 2 directions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x: (B, C, F', T')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = self.flatten(x)        # (B, T, C*F)
        output, _ = self.bilstm(x) # (B, T, 400)
        x = output.mean(dim=1)     # temporal average pooling
        return self.classifier(x)


# ===================================================
# ✅ 4️⃣ Focal Loss
# ===================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.nll_loss(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduce else F_loss


# ===================================================
# ✅ 5️⃣ Main 학습 루프
# ===================================================

def main():
    torch.manual_seed(Config.seed)
    device = torch.device(Config.device)

    # === 1) PhysioNet 레이블 ===
    labels_df = pd.read_csv(Config.physionet_label_file, names=['record', 'label'])
    labels_df = labels_df[labels_df['label'].isin(['N', 'AF', 'O'])]
    labels_df['label'] = labels_df['label'].map({'N': 0, 'AF': 1, 'O': 1})
    labels_df = labels_df.reset_index(drop=True)

    X_list, y_list = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        rec = wfdb.rdrecord(os.path.join(Config.physionet_data_dir, row['record']))
        raw_signal = rec.p_signal[:, 0]  # single-lead
        log_spec = compute_log_spectrogram(raw_signal)
        # reshape to (1, F, T)
        log_spec = np.expand_dims(log_spec, axis=0)
        X_list.append(log_spec)
        y_list.append(row['label'])

    # padding/truncate to same size along time axis
    max_time = max(x.shape[2] for x in X_list)
    X_pad = []
    for x in X_list:
        pad_width = max_time - x.shape[2]
        if pad_width > 0:
            x = np.pad(x, ((0,0),(0,0),(0,pad_width)), mode='constant')
        else:
            x = x[:,:,:max_time]
        X_pad.append(x)

    X = np.stack(X_pad)
    y = np.array(y_list)

    # split
    indices = np.arange(len(y))
    train_idx, temp_idx, _, temp_y = train_test_split(indices, y, test_size=0.2, stratify=y, random_state=Config.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_y, random_state=Config.seed)

    train_loader = DataLoader(ECGDatasetSpectrogram(X[train_idx], y[train_idx]), batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(ECGDatasetSpectrogram(X[val_idx], y[val_idx]), batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(ECGDatasetSpectrogram(X[test_idx], y[test_idx]), batch_size=Config.batch_size, shuffle=False)

    # === 2) 모델 ===
    model = CRNN().to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    modeltime = time.strftime('%m%d_%H%M%S')
    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = float('inf')
    early_stop_counter = 0

    # === 3) 학습 ===
    for epoch in tqdm(range(Config.num_epochs)):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += y_batch.size(0)
            correct_train += predicted.eq(y_batch).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"last.pth"))

        # === Validation ===
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                _, predicted = outputs.max(1)
                total_val += y_batch.size(0)
                correct_val += predicted.eq(y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pth"))
            min_val_loss = avg_val_loss
            print(f"✅ Saved best model to {checkpoint_dir}")
            early_stop_counter = 0
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= Config.patience:
        #         print("Early stopping triggered!")
        #         break

    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pth"))

    # === 4) Test ===
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best.pth")))
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # === Metrics ===
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float('nan')

    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    print(f"✅ Final Test F1-Score : {f1:.4f}")
    print(f"✅ Final Test ROC AUC  : {auc:.4f}")

    # output directory 생성
    output_dir = os.path.join('./output', modeltime)
    os.makedirs(output_dir, exist_ok=True)

    # === Classification report ===
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

    # === Confusion matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"./output/{modeltime}/confusion_matrix.png")
    plt.show()

    # === ROC curve ===
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"./output/{modeltime}/roc_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
