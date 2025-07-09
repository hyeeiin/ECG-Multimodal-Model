# dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class ECGMultimodalDataset(Dataset):
    def __init__(self, indices, labels_df, ecg_signals, clinical_df,
                 ecg_scaler=None, clinical_scaler=None, transform=None, config=None):
        self.transform = transform
        self.config = config

        # === Fold별 Subset ===
        self.labels_df = labels_df[labels_df['index'].isin(indices)].reset_index(drop=True)
        self.ecg_signals = ecg_signals.loc[ecg_signals.index.isin(indices)]
        self.clinical_df = clinical_df[clinical_df['index'].isin(indices)].reset_index(drop=True)

        self.ecg_scaler = ecg_scaler
        self.clinical_scaler = clinical_scaler

        # === Scale ===
        if self.ecg_scaler is not None:
            self.ecg_signals_scaled = pd.DataFrame(
                self.ecg_scaler.transform(self.ecg_signals),
                index=self.ecg_signals.index,
                columns=self.ecg_signals.columns
            )
        else:
            self.ecg_signals_scaled = self.ecg_signals

        if self.clinical_scaler is not None:
            clinical_numeric = self.clinical_df.drop(columns=['index'])
            self.clinical_scaled = pd.DataFrame(
                self.clinical_scaler.transform(clinical_numeric),
                index=self.clinical_df['index'],
                columns=clinical_numeric.columns
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
            self.config.image_dir, str(index), f"{str(index).zfill(3)}ECG_lead2.jpg"
        )
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        ecg_signal = torch.tensor(self.ecg_signals_scaled.loc[index].values, dtype=torch.float)
        clinical = torch.tensor(self.clinical_scaled.loc[index].values, dtype=torch.float)

        return image, ecg_signal, clinical, torch.tensor(label, dtype=torch.long)

def get_all_data(config):
    labels_df = pd.read_excel(config.label_file)
    clinical_df = pd.read_excel(config.clinical_file)
    clinical_df = clinical_df[['Wt','AGE','IDX']] # 상관 있는 변수만 사용
    ecg_signals = pd.read_csv(config.ecg_csv, index_col=0)

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

    print(f"✅ label indices: {len(label_indices)}")
    print(f"✅ ecg indices: {len(ecg_indices)}")
    print(f"✅ clinical indices: {len(clinical_indices)}")
    print(f"✅ image indices: {len(image_indices)}")
    print(f"✅ Final common indices: {len(common_indices)}")

    labels_df = labels_df[labels_df['index'].isin(common_indices)].reset_index(drop=True)
    ecg_signals = ecg_signals.loc[ecg_signals.index.isin(common_indices)]
    clinical_df = clinical_df[clinical_df['index'].isin(common_indices)].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    transform.config = config

    labels = labels_df['label'].values
    indices = np.arange(len(labels))

    return labels_df, ecg_signals, clinical_df, labels, indices, transform
