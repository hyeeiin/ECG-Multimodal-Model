# evaluation_signal.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import matplotlib.pyplot as plt

from config import Config  # ✅ 너의 Config: label_file, ecg_csv, trained_model_path 설정!

# === ✅ 1) PTB-XL 동일 전처리 ===
def remove_baseline_drift(signal, window_size=200):
    baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return signal - baseline

def lowpass_filter(signal, cutoff=40, fs=250, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def preprocess_signal(raw_signal):
    # Digitized ECG: 이미 250Hz라고 가정
    signal = remove_baseline_drift(raw_signal)
    signal = lowpass_filter(signal, cutoff=40, fs=250, order=5)
    if len(signal) > 2476:
        signal = signal[:2476]
    else:
        signal = np.pad(signal, (0, 2476 - len(signal)))
    return signal.copy()

# === ✅ 2) Digitized ECG Dataset ===
class DigitizedECGDataset(Dataset):
    def __init__(self, indices, labels_df, ecg_signals, ecg_scaler=None):
        self.labels_df = labels_df[labels_df['index'].isin(indices)].reset_index(drop=True)
        self.ecg_signals = ecg_signals.loc[ecg_signals.index.isin(indices)]
        self.ecg_scaler = ecg_scaler

        if self.ecg_scaler is not None:
            self.ecg_signals_scaled = pd.DataFrame(
                self.ecg_scaler.transform(self.ecg_signals),
                index=self.ecg_signals.index,
                columns=self.ecg_signals.columns
            )
        else:
            self.ecg_signals_scaled = self.ecg_signals

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        index = row['index']
        label = row['label']

        ecg_signal = self.ecg_signals_scaled.loc[index].values
        ecg_signal = preprocess_signal(ecg_signal)
        ecg_signal = torch.tensor(ecg_signal.copy(), dtype=torch.float).unsqueeze(0)
        return ecg_signal, torch.tensor(label, dtype=torch.long)

# === ✅ 3) Loader ===
def get_digitized_test_loader():
    labels_df = pd.read_excel(Config.label_file) if Config.label_file.endswith('.xlsx') else pd.read_csv(Config.label_file)
    ecg_signals = pd.read_csv(Config.ecg_csv, index_col=0)

    labels_df = labels_df[labels_df['label'] != 'Borderline']
    labels_df['label'] = labels_df['label'].map({'Normal': 0, 'Abnormal': 1})
    labels_df['index'] = labels_df['index'].astype(int)
    ecg_signals.index = ecg_signals.index.astype(int)

    common_indices = list(set(labels_df['index']) & set(ecg_signals.index))
    labels_df = labels_df[labels_df['index'].isin(common_indices)].reset_index(drop=True)
    ecg_signals = ecg_signals.loc[ecg_signals.index.isin(common_indices)]

    indices = labels_df.index.tolist()
    ecg_scaler = StandardScaler().fit(ecg_signals)

    test_ds = DigitizedECGDataset(indices, labels_df, ecg_signals, ecg_scaler)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    return test_loader

# === ✅ 4) ResNet1D_SE (PTB-XL 구조 동일)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
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

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
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

class ResNet1D_SE(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, base_filters=64):
        super().__init__()
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

# === ✅ 5) 평가 루프 ===
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_true, (np.array(y_prob) >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(scores)]
    return best_t

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_digitized_test_loader()

    model = ResNet1D_SE().to(device)
    model.load_state_dict(torch.load("./checkpoints/signal/0716_165523/best.pth"))
    model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing Digitized ECG"):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    best_t = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= best_t).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    acc = (y_true == y_pred).mean()

    print(f"\n✅ Best Threshold: {best_t:.2f}")
    print(f"✅ Test AUC: {auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Digitized ECG)")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Digitized ECG)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
