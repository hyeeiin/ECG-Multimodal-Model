# train_signal_only.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

from config import Config

# === 1️⃣ 전처리 함수 (signal_model.py에서 가져옴) ===
def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)

def remove_baseline_drift(signal, window_size=200):
    baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return signal - baseline

def lowpass_filter(signal, cutoff=0.05, fs=1.0, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def preprocess_signal(raw_signal):
    # signal = z_score_normalize(raw_signal)
    signal = remove_baseline_drift(raw_signal)
    signal = lowpass_filter(signal)
    return signal.copy()  # 연속적인 배열로 반환

# === 2️⃣ Signal-only Dataset ===
class SignalOnlyDataset(Dataset):
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
        ecg_signal = preprocess_signal(ecg_signal)  # signal_model.py 전처리 적용
        ecg_signal = np.copy(ecg_signal)  # 연속적인 배열로 복사
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float)
        return ecg_signal, torch.tensor(label, dtype=torch.long)

# === 3️⃣ Dataloader split ===
def get_signalonly_dataloaders():
    # labels_df = pd.read_excel(Config.af_label_file)
    labels_df = pd.read_excel(Config.arrhythmia_label_file)
    ecg_signals = pd.read_csv(Config.ecg_csv, index_col=0)

    labels_df['index'] = labels_df['index'].astype(int)
    ecg_signals.index = ecg_signals.index.astype(int)

    # # 1. AF(=1), Normal(=0) 필터링
    # labels_df = labels_df[(labels_df['label'] != 'Normal')]
    # labels_df['label'] = labels_df['label'].map({'Abnormal': 0, 'AF': 1, 'Borderline': 0})
    # 1. Arrhythmia(=1), Abnormal(=0) 필터링
    labels_df = labels_df[(labels_df['label'] != 'Normal') & (labels_df['label'] != 'Borderline')]
    labels_df['label'] = labels_df['label'].map({'Abnormal': 0, 'Arrhythmia': 1})

    # 2. 인덱스 정합성 체크
    valid_indices = set(labels_df['index']) & set(ecg_signals.index)
    labels_df = labels_df[labels_df['index'].isin(valid_indices)].reset_index(drop=True)
    ecg_signals = ecg_signals.loc[ecg_signals.index.isin(valid_indices)]

    # 3. 클래스별 분할
    af_df = labels_df[labels_df['label'] == 1]
    normal_df = labels_df[labels_df['label'] == 0]

    # assert len(af_df) == 6, f"AF 데이터는 6개여야 합니다. 현재: {len(af_df)}개"
    # assert len(normal_df) == 52, f"AF 데이터는 6개여야 합니다. 현재: {len(normal_df)}개"

    # # 4. AF: 2 train, 2 val, 2 test (총 6개라고 가정)
    # af_indices = af_df['index'].tolist()
    # np.random.seed(Config.seed)
    # np.random.shuffle(af_indices)
    # af_train = af_indices[:2]
    # af_test = af_indices[2:]

    # # 5. Normal: 200개 중 120 train, 40 val, 40 test
    # normal_indices = normal_df['index'].tolist()
    # np.random.shuffle(normal_indices)
    # normal_train = normal_indices[:68]
    # normal_val = normal_indices[68:90]
    # normal_test = normal_indices[90:]

    # # 6. 최종 인덱스
    # train_indices = af_train + normal_train
    # val_indices = normal_val
    # test_indices = af_test + normal_test

    # arrhythmia는 42개, abnormal은 52개로 꽤나 balanced
    indices = labels_df.index.tolist()
    labels = labels_df['label'].values
    
    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=Config.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=Config.seed
    )

    train_indices = labels_df.iloc[train_idx]['index'].tolist()
    val_indices = labels_df.iloc[val_idx]['index'].tolist()
    test_indices = labels_df.iloc[test_idx]['index'].tolist()

    # 7. 스케일러 학습은 train set으로만
    train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_indices)]
    ecg_scaler = StandardScaler().fit(train_ecg)

    train_ds = SignalOnlyDataset(train_indices, labels_df, ecg_signals, ecg_scaler)
    val_ds = SignalOnlyDataset(val_indices, labels_df, ecg_signals, ecg_scaler)
    test_ds = SignalOnlyDataset(test_indices, labels_df, ecg_signals, ecg_scaler)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    return train_loader, val_loader, test_loader


# === 4️⃣ 모델 (signal_model.py에서 가져옴) ===
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

# === 5️⃣ Focal Loss (signal_model.py에서 가져옴) ===
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

# === 6️⃣ Threshold 최적화 (signal_model.py에서 가져옴) ===
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_true, (np.array(y_prob) >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(scores)]
    return best_t

# === 7️⃣ Training ===
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Using device: {device}")

    train_loader, val_loader, test_loader = get_signalonly_dataloaders()

    model = ResNet1D_SE().to(device)
    # model.load_state_dict(torch.load("./checkpoints/signal/0718_115500/best.pth"))
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=30
    )

    modeltime = time.strftime('%m%d_%H%M%S')
    checkpoint_dir = os.path.join('./checkpoints', modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in tqdm(range(30)):
        print(f"\n=== Epoch [{epoch+1}/30] ===")
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for signals, labels in tqdm(train_loader, desc="Training"):
            model.train()
            signals, labels = signals.to(device), labels.to(device)
            signals = signals.unsqueeze(1)  # [B, L] → [B, 1, L]

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for signals, labels in tqdm(val_loader, desc="Validating"):
                signals, labels = signals.to(device), labels.to(device)
                signals = signals.unsqueeze(1)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val = predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"last.pth"))

        if avg_val_loss < min_val_loss:
            ckpt_path = os.path.join(checkpoint_dir, f"best_signal_only_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pth"))
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved best model to {ckpt_path}")
            min_val_loss = avg_val_loss
            early_stop_counter = 0
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= 5:  # Config.patience 대체
        #         print(f"⏹️ Early stopping at epoch {epoch+1}")
        #         break

    print("🎉 Training completed!")

    # === Test ===
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best.pth")))
    # model.load_state_dict(torch.load("./checkpoints/0714_095624/best.pth"))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)
            signals = signals.unsqueeze(1)

            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Metrics ===
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

    # output directory 생성
    output_dir = os.path.join('./output', modeltime)
    os.makedirs(output_dir, exist_ok=True)

    # === Classification report ===
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Abnormal', 'AF']))

    # === Confusion matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Abnormal', 'AF'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"./output/{modeltime}/confusion_matrix_best.png")
    plt.show()

    # === ROC curve ===
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"./output/{modeltime}/roc_curve_best.png")
    plt.show()

    #==========================LAST====================
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"last.pth")))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)
            signals = signals.unsqueeze(1)

            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Metrics ===
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

    # output directory 생성
    output_dir = os.path.join('./output', modeltime)
    os.makedirs(output_dir, exist_ok=True)

    # === Classification report ===
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Abnormal', 'AF']))

    # === Confusion matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Abnormal', 'AF'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"./output/{modeltime}/confusion_matrix.png")
    plt.show()

    # === ROC curve ===
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
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