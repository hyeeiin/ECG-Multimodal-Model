# dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class ECGMultimodalDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform

        # === Load ===
        self.labels_df = pd.read_excel(config.label_file)
        self.clinical_df = pd.read_excel(config.clinical_file)
        self.ecg_signals = pd.read_csv(config.ecg_csv, index_col=0)

        # === Remove borderline ===
        self.labels_df = self.labels_df[self.labels_df['label'] != 'Borderline']
        self.labels_df['label'] = self.labels_df['label'].map({'Normal': 0, 'Abnormal': 1})

        # === IDX 컬럼 rename ===
        if 'IDX' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.rename(columns={'IDX': 'index'})

        # === Type 안전 ===
        self.labels_df['index'] = self.labels_df['index'].astype(int)
        self.clinical_df['index'] = self.clinical_df['index'].astype(int)
        self.ecg_signals.index = self.ecg_signals.index.astype(int)

        # === 이미지 폴더 확인 ===
        image_indices = set()
        for folder in os.listdir(config.image_dir):
            if folder.isdigit():
                image_indices.add(int(folder))

        # === 빠진 index 처리 ===
        known_missing = {17, 23, 36, 43, 51, 62, 115, 158}
        image_indices -= known_missing  # 혹시 포함돼있으면

        # === 교집합 ===
        label_indices = set(self.labels_df['index'])
        ecg_indices = set(self.ecg_signals.index)
        clinical_indices = set(self.clinical_df['index'])

        common_indices = label_indices & ecg_indices & clinical_indices & image_indices

        print(f"✅ label indices: {len(label_indices)}")
        print(f"✅ ecg indices: {len(ecg_indices)}")
        print(f"✅ clinical indices: {len(clinical_indices)}")
        print(f"✅ image indices: {len(image_indices)}")
        print(f"✅ Final common indices: {len(common_indices)}")

        if len(common_indices) == 0:
            raise ValueError("❌ 교집합 인덱스가 0개입니다. index 컬럼, image 폴더, csv index_col 확인하세요!")

        # === Filter ===
        self.labels_df = self.labels_df[self.labels_df['index'].isin(common_indices)].reset_index(drop=True)
        self.ecg_signals = self.ecg_signals.loc[self.ecg_signals.index.isin(common_indices)]
        self.clinical_df = self.clinical_df[self.clinical_df['index'].isin(common_indices)].reset_index(drop=True)

        # === Scale ===
        self.ecg_scaler = StandardScaler()
        self.ecg_signals_scaled = pd.DataFrame(
            self.ecg_scaler.fit_transform(self.ecg_signals),
            index=self.ecg_signals.index,
            columns=self.ecg_signals.columns
        )

        clinical_numeric = self.clinical_df.drop(columns=['index'])
        self.clinical_scaler = StandardScaler()
        self.clinical_scaled = pd.DataFrame(
            self.clinical_scaler.fit_transform(clinical_numeric),
            index=self.clinical_df['index'],
            columns=clinical_numeric.columns
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        index = row['index']
        label = row['label']

        # === 이미지 ===
        img_path = os.path.join(
            self.config.image_dir, str(index), f"{str(index).zfill(3)}ECG_lead2.jpg"
        )
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # === 시계열 ===
        ecg_signal = torch.tensor(self.ecg_signals_scaled.loc[index].values, dtype=torch.float)

        # === 임상 ===
        clinical = torch.tensor(self.clinical_scaled.loc[index].values, dtype=torch.float)

        return image, ecg_signal, clinical, torch.tensor(label, dtype=torch.long)


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = ECGMultimodalDataset(config, transform=transform)

    labels = dataset.labels_df['label'].values
    indices = np.arange(len(labels))

    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=config.seed
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=config.seed
    )

    print(f"🔍 Split: Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
