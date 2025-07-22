# train_signal_only_ptb.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from scipy.signal import butter, filtfilt
import wfdb

# === 1️⃣ 전처리 ===
def remove_baseline_drift(signal, window_size=200):
    baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return signal - baseline

def lowpass_filter(signal, cutoff=40, fs=250, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

# === 2️⃣ Dataset ===
# 1 LEAD
class PTBXLSignalDataset(Dataset):
    def __init__(self, records, labels, length=2476):
        self.records = records
        self.labels = labels
        self.length = length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        label = self.labels[idx]
        signal, _ = wfdb.rdsamp(record_path, channels=[1])  # Lead II
        signal = signal.flatten()
        signal = signal[::2]  # 500Hz → 250Hz
        signal = remove_baseline_drift(signal)
        signal = lowpass_filter(signal)
        if len(signal) > self.length:
            signal = signal[:self.length]
        else:
            pad = np.zeros(self.length - len(signal))
            signal = np.concatenate([signal, pad])
        return torch.tensor(signal.copy(), dtype=torch.float).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# 12 LEAD
# class PTBXLSignalDataset(Dataset):
#     def __init__(self, records, labels, length=2476):
#         self.records = records
#         self.labels = labels
#         self.length = length

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         record_path = self.records[idx]
#         label = self.labels[idx]

#         # ✅ 전체 12개 리드 불러오기
#         signal, _ = wfdb.rdsamp(record_path)  # [time, 12]
#         signal = signal.T  # [12, time]

#         # ✅ 전체 리드 전처리
#         processed = []
#         for lead in signal:
#             lead = lead[::2]  # 500Hz → 250Hz
#             lead = remove_baseline_drift(lead)
#             lead = lowpass_filter(lead)
#             if len(lead) > self.length:
#                 lead = lead[:self.length]
#             else:
#                 pad = np.zeros(self.length - len(lead))
#                 lead = np.concatenate([lead, pad])
#             processed.append(lead)

#         signal_tensor = torch.tensor(np.array(processed).copy(), dtype=torch.float)  # [12, 2476]
#         return signal_tensor, torch.tensor(label, dtype=torch.long)

# === 3️⃣ 모델 ===
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_true, (np.array(y_prob) >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(scores)]
    return best_t

# === 4️⃣ 메인 ===
def main():
    modeltime = time.strftime('%m%d_%H%M%S')
    checkpoint_dir = os.path.join('./checkpoints/signal', modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_csv('./data/ptb_xl/ptbxl_database.csv')
    df['scp_codes'] = df['scp_codes'].apply(eval)
    # Normal vs. Abnormal
    # df['label'] = df['scp_codes'].apply(
    #     lambda x: 0 if ('NORM' in x and float(x['NORM']) == 100.0) else 1
    # )
    # AF (심방세동) - Abnormal
    abnormal_rhythm_codes = ['SR', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']

    # 새로운 라벨링 로직
    def label_ecg(scp_code_str):
        try:
            # scp_dict = eval(scp_code_str)
            scp_dict = scp_code_str
            if ('AFIB' in scp_dict) and float(scp_dict['AFIB'] == 100.0):
                return 1  # AFIB
            elif any(code in scp_dict and float(scp_dict[code]) == 100.0 for code in abnormal_rhythm_codes):
                return 0  # Other abnormal rhythm
            else:
                return 2  # Other / ignore
        except:
            return 2  # malformed or unrecognized
    # df['label'] = df['scp_codes'].apply(
    #     lambda x: 0 if ('NORM' in x and float(x['NORM']) == 100.0)
    #     else (1 if ('AFIB' in x and float(x['AFIB']) == 100.0)
    #         else 2)
    # )
    # df = df[df['label'] != 2].reset_index(drop=True)
    df['label'] = df['scp_codes'].apply(label_ecg)

    # 2️⃣ 나머지 라벨 제거
    df = df[df['label'] != 2].reset_index(drop=True)

    # # 3️⃣ 클래스 별 분리
    # df_afib = df[df['label'] == 1]
    # df_norm = df[df['label'] == 0].sample(n=200, random_state=42)

    # # 4️⃣ 다시 합치기
    # df = pd.concat([df_afib, df_norm]).sample(frac=1, random_state=42).reset_index(drop=True)

    records = ['./data/ptb_xl/' + r for r in df['filename_hr']]
    labels = df['label'].values

    # === Split
    X_train, X_temp, y_train, y_temp = train_test_split(records, labels, test_size=0.4, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # === Weighted Sampler
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_ds = PTBXLSignalDataset(X_train, y_train)
    val_ds = PTBXLSignalDataset(X_val, y_val)
    test_ds = PTBXLSignalDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # === Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D_SE().to(device)
    # model = ResNet1D_SE(input_channels=12).to(device)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=30)

    best_val_loss = float('inf')
    for epoch in tqdm(range(10)):
        model.train()
        train_loss = 0.0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # === Validation
        model.eval()
        val_loss = 0.0
        y_true, y_prob = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(probs)
        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(y_true, y_prob)
        print(f"✅ Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_val_loss = avg_val_loss

    # === Test ===
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best.pth')))
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    best_t = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= best_t).astype(int)

    print(f"✅ Best threshold: {best_t:.2f}")
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    print(f"✅ Final Test F1-Score : {f1:.4f}")
    print(f"✅ Final Test ROC AUC  : {auc:.4f}")

    output_dir = os.path.join('./output', modeltime)
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Classification Report ===")
    # print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
    print(classification_report(y_true, y_pred, target_names=['Normal', 'AF']))

    cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'AF'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"./output/{modeltime}/confusion_matrix_best.png")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"./output/{modeltime}/roc_curve_best.png")
    plt.show()

if __name__ == "__main__":
    main()
