# train_image_only.py

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve

from config import Config

# === 1️⃣ Image-only Dataset ===
class ImageOnlyDataset(Dataset):
    def __init__(self, indices, labels_df, transform=None):
        self.transform = transform
        self.labels_df = labels_df[labels_df['index'].isin(indices)].reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        index = row['index']
        label = row['label']

        img_path = os.path.join(
            Config.image_dir, str(index), f"{str(index).zfill(3)}ECG_lead2.jpg"
        )
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# === 2️⃣ Dataloader split ===
def get_imageonly_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((Config.img_height, Config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    labels_df = pd.read_excel(Config.label_file)
    labels_df = labels_df[labels_df['label'] != 'Borderline']
    labels_df['label'] = labels_df['label'].map({'Normal': 0, 'Abnormal': 1})
    labels_df['index'] = labels_df['index'].astype(int)

    image_indices = set(int(folder) for folder in os.listdir(Config.image_dir) if folder.isdigit())
    known_missing = {17, 23, 36, 43, 51, 62, 115, 158}
    image_indices -= known_missing

    label_indices = set(labels_df['index'])
    common_indices = label_indices & image_indices

    labels_df = labels_df[labels_df['index'].isin(common_indices)].reset_index(drop=True)
    indices = labels_df.index.tolist()
    labels = labels_df['label'].values

    from sklearn.model_selection import train_test_split

    train_idx, temp_idx, _, temp_y = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=Config.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=Config.seed
    )

    train_ds = ImageOnlyDataset(labels_df.iloc[train_idx]['index'].tolist(), labels_df, transform)
    val_ds = ImageOnlyDataset(labels_df.iloc[val_idx]['index'].tolist(), labels_df, transform)
    test_ds = ImageOnlyDataset(labels_df.iloc[test_idx]['index'].tolist(), labels_df, transform)

    test_indices = labels_df.iloc[test_idx]['index'].tolist()
    val_indices = labels_df.iloc[val_idx]['index'].tolist()
    print(f"val indices: {val_indices}")
    print(f"test indices: {test_indices}")

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# === 3️⃣ Image-only Classifier ===
class ImageOnlyClassifier(nn.Module):
    def __init__(self):
        super(ImageOnlyClassifier, self).__init__()
        self.image_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, Config.num_classes)

    def forward(self, x):
        return self.image_encoder(x)

# === 4️⃣ Training ===
def main():
    torch.manual_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"📍 Using device: {device}")

    train_loader, val_loader, test_loader = get_imageonly_dataloaders()

    model = ImageOnlyClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.lr)

    modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in tqdm(range(Config.num_epochs)):
        print(f"\n=== Epoch [{epoch+1}/{Config.num_epochs}] ===")
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"last.pth"))

        if avg_val_loss < min_val_loss:
            ckpt_path = os.path.join(checkpoint_dir, f"best_image_only_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pth"))
            print(f"✅ Saved best model to {ckpt_path}")
            min_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    print("🎉 Training completed!")

    # === Test ===
    # Test loop
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"last.pth")))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images= images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 확률만
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Metrics ===
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
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
    plt.savefig(f"./output/{modeltime}/confusion_matrix.png")
    plt.show()

    # === ROC curve ===
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

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

    # Test loop
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best.pth")))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images= images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 확률만
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Metrics ===
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
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
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

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

if __name__ == "__main__":
    main()
