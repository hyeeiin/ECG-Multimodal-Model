# dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.signal import butter, filtfilt
from pickle import dump

class ECGMultimodalDataset(Dataset):
    def __init__(self, indices, labels_df, ecg_signals, clinical_df,
                 ecg_scaler=None, clinical_scaler=None, transform=None):
        self.transform = transform

        self.labels_df = labels_df[labels_df['index'].isin(indices)].reset_index(drop=True)
        self.ecg_signals = ecg_signals.loc[ecg_signals.index.isin(indices)]
        self.clinical_df = clinical_df[clinical_df['index'].isin(indices)].reset_index(drop=True)

        self.ecg_scaler = ecg_scaler
        self.clinical_scaler = clinical_scaler

        # numerical column 정의
        self.clinical_numeric_scaler_cols = ["AGE","Wt"]

        if self.ecg_scaler is not None:
            self.ecg_signals_scaled = pd.DataFrame(
                self.ecg_scaler.transform(self.ecg_signals),
                index=self.ecg_signals.index,
                columns=self.ecg_signals.columns
            )
        else:
            self.ecg_signals_scaled = self.ecg_signals
            print("None")

        if self.clinical_scaler is not None:
            scaled_numeric = self.clinical_df[self.clinical_numeric_scaler_cols]
            self.clinical_scaled = pd.DataFrame(
                self.clinical_scaler.transform(scaled_numeric),
                index=self.clinical_df['index'],
                columns=self.clinical_numeric_scaler_cols
            )
        else:
            self.clinical_scaled = self.clinical_df.drop(columns=['index'])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        index = row['index']
        label = row['label']

        img_path = os.path.join(
            self.transform.config.image_dir, str(index), f"{str(index).zfill(3)}ECG_lead2.jpg"
        )
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        ecg_signal = self.ecg_signals_scaled.loc[index].values
        ecg_signal = self.preprocess_signal(ecg_signal)  # ✅ 샘플별로 적용
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float)
        clinical = torch.tensor(self.clinical_scaled.loc[index].values, dtype=torch.float)

        return image, ecg_signal, clinical, torch.tensor(label, dtype=torch.long)
    # === 1️⃣ 전처리 함수 (signal_model.py에서 가져옴) ===
    def z_score_normalize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)

    def remove_baseline_drift(self, signal, window_size=200):
        baseline = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        return signal - baseline

    def lowpass_filter(self, signal, cutoff=0.05, fs=1.0, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def preprocess_signal(self, raw_signal):
        # signal = z_score_normalize(raw_signal)
        signal = self.remove_baseline_drift(raw_signal)
        signal = self.lowpass_filter(signal)
        return signal.copy()  # 연속적인 배열로 반환

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    transform.config = config  # 이미지 경로를 위해 config 참조

    # === Load data ===
    labels_df = pd.read_excel(config.label_file)
    clinical_df = pd.read_csv(config.clinical_file)
    clinical_df = clinical_df.drop("ECG", axis=1) # 상관 있는 변수만 사용 -> 전체 변수 사용
    ecg_signals = pd.read_csv(config.ecg_csv, index_col=0)

    # === Preprocessing ===
    labels_df = labels_df[labels_df['label'] != 'Borderline']
    labels_df['label'] = labels_df['label'].map({'Normal': 0, 'Abnormal': 1})
    # labels_df['label'] = labels_df['label'].map({'Normal': 0, 'AF': 1})

    if 'IDX' in clinical_df.columns:
        clinical_df = clinical_df.rename(columns={'IDX': 'index'})

    labels_df['index'] = labels_df['index'].astype(int)
    clinical_df['index'] = clinical_df['index'].astype(int)
    ecg_signals.index = ecg_signals.index.astype(int)

    image_indices = set(int(folder) for folder in os.listdir(config.image_dir) if folder.isdigit())
    known_missing = {17, 23, 36, 43, 51, 62, 115, 158}
    image_indices -= known_missing

    label_indices = set(labels_df['index'])
    ecg_indices = set(ecg_signals.index)
    clinical_indices = set(clinical_df['index'])

    common_indices = label_indices & ecg_indices & clinical_indices & image_indices

    print(f"✅ label indices: {len(label_indices)}")
    print(f"✅ ecg indices: {len(ecg_indices)}")
    print(f"✅ clinical indices: {len(clinical_indices)}")
    print(f"✅ image indices: {len(image_indices)}")
    print(f"✅ Final common indices: {len(common_indices)}")

    labels_df = labels_df[labels_df['index'].isin(common_indices)].reset_index(drop=True)
    ecg_signals = ecg_signals.loc[ecg_signals.index.isin(common_indices)]
    clinical_df = clinical_df[clinical_df['index'].isin(common_indices)].reset_index(drop=True)

    labels = labels_df['label'].values
    indices = np.arange(len(labels))

    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=config.seed
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=config.seed
    )

    # # ✅ 전체의 30%가 temp → 1/3 val = 10%, 2/3 test = 20%
    # train_idx, temp_idx, _, temp_y = train_test_split(
    #     indices, labels, test_size=0.3, stratify=labels, random_state=config.seed
    # )

    # val_idx, test_idx = train_test_split(
    #     temp_idx, test_size=(2/3), stratify=temp_y, random_state=config.seed
    # )


    print(f"🔍 Split: Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

    train_indices = labels_df.iloc[train_idx]['index'].tolist()
    val_indices = labels_df.iloc[val_idx]['index'].tolist()
    test_indices = labels_df.iloc[test_idx]['index'].tolist()

    print(f"val indeices: {val_indices}")
    print(f"test indices: {test_indices}")

    train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_indices)]
    ecg_scaler = StandardScaler().fit(train_ecg)

    # train_clinical = clinical_df[clinical_df['index'].isin(train_indices)].drop(columns=['index'])
    clinical_numeric_scaler_cols = ["AGE","Wt"]
    train_clinical = clinical_df[clinical_df['index'].isin(train_indices)][clinical_numeric_scaler_cols]
    clinical_scaler = StandardScaler().fit(train_clinical)

    train_ds = ECGMultimodalDataset(train_indices, labels_df, ecg_signals, clinical_df,
                                     ecg_scaler, clinical_scaler, transform)
    val_ds = ECGMultimodalDataset(val_indices, labels_df, ecg_signals, clinical_df,
                                   ecg_scaler, clinical_scaler, transform)
    test_ds = ECGMultimodalDataset(test_indices, labels_df, ecg_signals, clinical_df,
                                    ecg_scaler, clinical_scaler, transform)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def get_testloader(config, test_indices):
    """
    ✅ test_indices: 네가 직접 지정한 index 리스트
    ✅ train/val split 없이 scaler는 train_loader에서 저장한 scaler로 사용하면 좋음.
       여기선 편의상 test 데이터만 로딩하도록 예시로 작성!
    """

    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    transform.config = config

    # === Load raw data ===
    labels_df = pd.read_excel(config.label_file)
    clinical_df = pd.read_excel(config.clinical_file)
    clinical_df = clinical_df[['Wt','AGE','IDX']]  # 필요한 컬럼만
    ecg_signals = pd.read_csv(config.ecg_csv, index_col=0)

    # === Preprocessing ===
    labels_df = labels_df[labels_df['label'] != 'Borderline']
    labels_df['label'] = labels_df['label'].map({'Normal': 0, 'Abnormal': 1})

    if 'IDX' in clinical_df.columns:
        clinical_df = clinical_df.rename(columns={'IDX': 'index'})

    labels_df['index'] = labels_df['index'].astype(int)
    clinical_df['index'] = clinical_df['index'].astype(int)
    ecg_signals.index = ecg_signals.index.astype(int)

    image_indices = set(int(folder) for folder in os.listdir(config.image_dir) if folder.isdigit())
    known_missing = {17, 23, 36, 43, 51, 62, 115, 158}
    image_indices -= known_missing

    label_indices = set(labels_df['index'])
    ecg_indices = set(ecg_signals.index)
    clinical_indices = set(clinical_df['index'])

    common_indices = label_indices & ecg_indices & clinical_indices & image_indices

    # ✅ 주어진 test_indices만 유효한 것으로 필터링
    valid_test_indices = list(set(test_indices) & common_indices)
    print(f"✅ Valid Test Indices: {valid_test_indices}")

    # === 스케일러: 보통은 train에서 가져와야 안전, 여기선 fit 예시로 보여줌 ===
    train_like_ecg = ecg_signals.loc[ecg_signals.index.isin(valid_test_indices)]
    ecg_scaler = StandardScaler().fit(train_like_ecg)

    train_like_clinical = clinical_df[clinical_df['index'].isin(valid_test_indices)].drop(columns=['index'])
    clinical_scaler = StandardScaler().fit(train_like_clinical)

    test_ds = ECGMultimodalDataset(valid_test_indices, labels_df, ecg_signals, clinical_df,
                                   ecg_scaler, clinical_scaler, transform)

    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return test_loader
