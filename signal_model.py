# ✅ ResNet1D + SEBlock 기반 ECG 분류 모델 (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset

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

# ✅ Focal Loss 정의
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

# ✅ Dataset 클래스
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ Threshold 최적화
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = [f1_score(y_true, (np.array(y_prob) >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(scores)]
    return best_t

# ✅ 평가 함수 (단일 모델)
def evaluate_model(model, test_loader, threshold=0.3):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            prob = F.softmax(output, dim=1)[:, 1]
            y_true.extend(y_batch.numpy())
            y_prob.extend(prob.numpy())
    threshold = find_best_threshold(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    print(f"✅ Best threshold: {threshold:.2f}")

    print("\n📊 [Evaluation Results - Single Model]")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"], zero_division=0))
    print(f"🔹 Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"🔹 F1-score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"🔹 AUC: {roc_auc_score(y_true, y_prob):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ✅ 학습 함수 (OneCycleLR 적용)
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )
    min_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"📍 Epoch {epoch+1}/{epochs} 완료")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < min_val_loss:
            print(f"Epoch {epoch} : loss decreased")
            min_val_loss = avg_val_loss

# ✅ 실행
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import os
    from glob import glob
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    csv_files = glob(r"./data/signals/*.csv")
    label_path = r"./data/labels.xlsx"

    label_df = pd.read_excel(label_path)
    label_map = {"Normal": 0, "Abnormal": 1}
    label_dict = label_df.set_index("index")["label"].map(label_map).to_dict()

    def z_score_normalize(signal):
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)

    def remove_baseline_drift(signal, window_size=200):
        baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        return signal - baseline

    def lowpass_filter(signal, cutoff=0.05, fs=1.0, order=5):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def preprocess_signal(raw_signal):
        signal = z_score_normalize(raw_signal)
        signal = remove_baseline_drift(signal)
        signal = lowpass_filter(signal)
        return signal

    X_list, y_list = [], []
    X_train_man, y_train_man, X_val_man, y_val_man, X_test_man, y_test_man = [], [], [], [], [], []
    test_idx = [122, 200, 161, 130, 211, 3, 232, 14, 190, 175, 251, 68, 160, 102, 168, 165, 237, 7, 129, 228, 180, 133]
    val_idx = [54, 116, 44, 27, 119, 48, 81, 205, 236, 30, 98, 41, 83, 6, 212, 204, 1, 45, 210, 110, 181, 156]
    for file_path in sorted(csv_files, key=lambda x: int(os.path.basename(x).replace(".csv", ""))):
        try:
            idx = int(os.path.basename(file_path).replace(".csv", ""))
            df = pd.read_csv(file_path, header=None)
            if df.empty:
                continue
            values = df.iloc[:, 0].to_numpy(dtype=np.float32)
            values = preprocess_signal(values)
            label = label_dict.get(idx, None)
            if label is None or pd.isna(label):
                continue
            X_list.append(values)
            y_list.append(label)


            # 수동 split
            if idx in test_idx:
                X_test_man.append(values)
                y_test_man.append(label)
            elif idx in val_idx:
                X_val_man.append(values)
                y_val_man.append(label)
            else:
                X_train_man.append(values)
                y_train_man.append(label)
        except:
            continue

    max_len = max(len(x) for x in X_list)
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    X = np.expand_dims(X_padded, axis=1)
    y = np.array(y_list)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ 8:1:1 split + train+val 합치기
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    X_train_final = np.concatenate([X_train, X_val])
    y_train_final = np.concatenate([y_train, y_val])

    X_train_padded = pad_sequences(X_train_man, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    X_val_padded = pad_sequences(X_val_man, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    X_test_padded = pad_sequences(X_test_man, maxlen=max_len, dtype="float32", padding="post", truncating="post")

    X_train = np.expand_dims(X_train_padded, axis=1)
    y_train = np.array(y_train_man)
    X_val = np.expand_dims(X_val_padded, axis=1)
    y_val = np.array(y_val_man)
    X_test = np.expand_dims(X_test_padded, axis=1)
    y_test = np.array(y_test_man)

    print("train", len(X_train))
    print("test", len(X_test))

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


    # 데이터 로더 준비
    model = ResNet1D_SE(input_channels=1)
    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=16, shuffle=True)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=16)
    train_model(model, train_loader, val_loader, epochs=30)

    evaluate_model(model, test_loader)