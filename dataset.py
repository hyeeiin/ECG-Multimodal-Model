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

        # --- Load data ---
        self.labels_df = pd.read_excel(config.label_file)
        self.clinical_df = pd.read_excel(config.clinical_file)
        self.ecg_signals = pd.read_csv(config.ecg_csv, index_col=0)

        # --- Filter out 'borderline' labels ---
        self.labels_df = self.labels_df[self.labels_df['Label'] != 'Borderline']
        self.labels_df['Label'] = self.labels_df['Label'].map({'Normal': 0, 'Abnormal': 1})

        # --- Valid indices only ---
        valid_indices = [i for i in range(1, 253) if i not in [17, 23, 36, 43, 51, 62, 115, 158]]
        self.labels_df = self.labels_df[self.labels_df['IDX'].isin(valid_indices)]

        # --- Scale ECG signals ---
        self.ecg_scaler = StandardScaler()
        self.ecg_signals_scaled = pd.DataFrame(
            self.ecg_scaler.fit_transform(self.ecg_signals),
            index=self.ecg_signals.index,
            columns=self.ecg_signals.columns
        )

        # --- Scale Clinical data ---
        clinical_numeric = self.clinical_df.drop(columns=['IDX'])
        self.clinical_scaler = StandardScaler()
        self.clinical_scaled = pd.DataFrame(
            self.clinical_scaler.fit_transform(clinical_numeric),
            index=self.clinical_df['IDX'],
            columns=clinical_numeric.columns
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        index = row['IDX']
        label = row['Label']

        # --- Load ECG image ---
        img_path = os.path.join(self.config.image_dir, str(index), f"{str(index).zfill(3)}ECG_lead2.jpg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # --- ECG signal ---
        ecg_signal = torch.tensor(self.ecg_signals_scaled.loc[index].values, dtype=torch.float)

        # --- Clinical features ---
        clinical = torch.tensor(self.clinical_scaled.loc[index].values, dtype=torch.float)

        return image, ecg_signal, clinical, torch.tensor(label, dtype=torch.long)


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    full_dataset = ECGMultimodalDataset(config, transform=transform)

    # === 안전하게 labels 추출 ===
    labels_df = full_dataset.labels_df.reset_index(drop=True)
    labels = labels_df['Label'].values

    # NaN 확인 후 삭제
    if pd.isnull(labels).any():
        print(labels_df)
        raise ValueError("🚨 labels에 NaN이 포함되어 있습니다. labels_df 내용을 다시 확인하세요!")

    indices = np.arange(len(labels))

    # === Stratified split ===
    from sklearn.model_selection import train_test_split

    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=config.seed)

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=config.seed)

    print(f"🔍 Stratified split → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)
    test_ds = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader