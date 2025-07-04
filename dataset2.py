# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image

class MultiModalDataset(Dataset):
    def __init__(self, image_paths, clinical_df, ecg_df, labels, transform=None):
        self.image_paths = image_paths
        self.clinical_df = clinical_df
        self.ecg_df = ecg_df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Load clinical data
        clinical_data = torch.tensor(self.clinical_df.iloc[idx].values, dtype=torch.float32)
        
        # Load ECG time-series
        ecg_data = torch.tensor(self.ecg_df.iloc[idx].values, dtype=torch.float32)
        
        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, clinical_data, ecg_data, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_dataloaders(config, clinical_df, ecg_df, image_paths, labels):
    dataset = MultiModalDataset(image_paths, clinical_df, ecg_df, labels, transform=get_transforms())
    val_size = int(len(dataset) * config.VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader
