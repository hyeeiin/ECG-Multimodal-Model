# train_signal_only.py (Updated for PhysioNet 2017 dataset)

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
import wfdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import Config

# === 1️⃣ 전처리 함수 (PhysioNet 스타일로 수정) ===
def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)

def bandpass_filter(signal, lowcut=16, highcut=149, fs=300, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def resample_signal(signal, orig_fs, target_fs=300):
    if orig_fs != target_fs:
        num_samples = int(len(signal) * (target_fs / orig_fs))
        signal = resample(signal, num_samples)
    return signal

def preprocess_signal(raw_signal, orig_fs=300, target_fs=300):
    signal = bandpass_filter(raw_signal, fs=orig_fs)
    signal = resample_signal(signal, orig_fs, target_fs)
    signal = z_score_normalize(signal)
    return signal

def augment_signal(signal):
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.01, signal.shape)
        signal += noise

    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        signal *= scale

    if np.random.rand() < 0.5:
        shift = np.random.randint(-10, 10)
        signal = np.roll(signal, shift)

    return signal.copy()

# === 2️⃣ Signal-only Dataset (PhysioNet용으로 수정) ===
class SignalOnlyDataset(Dataset):
    def __init__(self, indices, labels_df, ecg_signals, augment=True, max_len=3000, split= 'train'):
        self.labels_df = labels_df.iloc[indices].reset_index(drop=True)
        self.ecg_signals = [ecg_signals[i] for i in indices]
        self.max_len = max_len
        self.augment = augment
        self.split = split

        # Pad sequences to max_len
        self.ecg_signals_padded = pad_sequences(
            self.ecg_signals, maxlen=max_len, dtype='float32', padding='post', truncating='post'
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        ecg_signal = self.ecg_signals_padded[idx]
        ecg_signal = preprocess_signal(ecg_signal, orig_fs=300, target_fs=300)
        if self.augment and self.split == 'train':
            ecg_signal = augment_signal(ecg_signal)
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float)
        label = self.labels_df.iloc[idx]['label']
        return ecg_signal, torch.tensor(label, dtype=torch.long)

# === 3️⃣ Dataloader split (PhysioNet 데이터 로드) ===
def get_signalonly_dataloaders():
    # PhysioNet 레이블 로드
    labels_df = pd.read_csv(Config.physionet_label_file, names=['record', 'label'])
    labels_df = labels_df[labels_df['label'].isin(['N', 'AF', 'O'])]
    labels_df['label'] = labels_df['label'].map({'N': 0, 'AF': 1, 'O': 1})
    labels_df = labels_df.reset_index(drop=True)

    # ECG 신호 로드
    ecg_signals = []
    valid_indices = []
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Loading PhysioNet data"):
        try:
            rec = wfdb.rdrecord(os.path.join(Config.physionet_data_dir, row['record']))
            raw_signal = rec.p_signal[:, 0]  # single-lead
            ecg_signals.append(raw_signal)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to load record {row['record']}: {e}")
            continue

    labels_df = labels_df.iloc[valid_indices].reset_index(drop=True)

    # 데이터 분할
    indices = np.arange(len(labels_df))
    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels_df['label'].values, test_size=0.2, stratify=labels_df['label'].values, random_state=Config.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=Config.seed
    )

    print(f"val indices: {val_idx}")
    print(f"test indices: {test_idx}")

    # Dataset 및 DataLoader 생성
    train_ds = SignalOnlyDataset(train_idx, labels_df, ecg_signals, split='train')
    val_ds = SignalOnlyDataset(val_idx, labels_df, ecg_signals, split='val')
    test_ds = SignalOnlyDataset(test_idx, labels_df, ecg_signals, split='test')

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    return train_loader, val_loader, test_loader

# === 4️⃣ 모델 (그대로 유지) ===
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

class ECGTransformer1D(nn.Module):
    def __init__(self, input_dim=1, seq_len=3000, num_classes=2,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(ECGTransformer1D, self).__init__()
        
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: [B, 1, L]
        x = self.conv(x)  # [B, d_model, L]
        x = x.permute(0, 2, 1)  # [B, L, d_model]
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer_encoder(x)  # [B, L, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, L]
        x = self.global_pool(x)
        return self.classifier(x)

# === 5️⃣ Focal Loss (그대로 유지) ===
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

# === 6️⃣ Threshold 최적화 (그대로 유지) ===
# def find_best_threshold(y_true, y_prob):
#     thresholds = np.arange(0.1, 0.9, 0.05)
#     scores = [f1_score(y_true, (np.array(y_prob) >= t).astype(int)) for t in thresholds]
#     best_t = thresholds[np.argmax(scores)]
#     return best_t

# === 7️⃣ Training (그대로 유지, DataLoader만 PhysioNet용으로 교체) ===
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Using device: {device}")

    train_loader, val_loader, test_loader = get_signalonly_dataloaders()

    model = ResNet1D_SE().to(device)
    # model = ECGTransformer1D().to(device)
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

    print("🎉 Training completed!")

    # === Test (Best model) ===
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best.pth")))
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
    # best_t = find_best_threshold(y_true, y_prob)
    # y_pred = (y_prob >= best_t).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    # print(f"✅ Best threshold: {best_t:.2f}")
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
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

    # === Confusion matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
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

    # === Test (Last model) ===
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
    # best_t = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # print(f"✅ Best threshold: {best_t:.2f}")
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    print(f"\n✅ Final Test Accuracy: {acc:.4f}")
    print(f"✅ Final Test F1-Score : {f1:.4f}")
    print(f"✅ Final Test ROC AUC  : {auc:.4f}")

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