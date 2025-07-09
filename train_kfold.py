# train.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import Config
from dataset_kfold import get_all_data, ECGMultimodalDataset
from multimodal import ECGMultimodalModel

def train_inner(inner_train_idx, inner_val_idx, labels_df, ecg_signals, clinical_df, transform, fold_dir):
    # === Split ===
    train_indices = labels_df.iloc[inner_train_idx]['index'].tolist()
    val_indices = labels_df.iloc[inner_val_idx]['index'].tolist()

    # === Scaler ===
    train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_indices)]
    ecg_scaler = StandardScaler().fit(train_ecg)
    train_clinical = clinical_df[clinical_df['index'].isin(train_indices)].drop(columns=['index'])
    clinical_scaler = StandardScaler().fit(train_clinical)

    # === Dataset & Loader ===
    train_ds = ECGMultimodalDataset(train_indices, labels_df, ecg_signals, clinical_df,
                                    ecg_scaler, clinical_scaler, transform, config=Config)
    val_ds = ECGMultimodalDataset(val_indices, labels_df, ecg_signals, clinical_df,
                                  ecg_scaler, clinical_scaler, transform, config=Config)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    # === Model ===
    model = ECGMultimodalModel(Config).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.lr)

    writer = SummaryWriter(os.path.join(fold_dir, "inner_logs"))
    min_val_loss = float('inf')
    early_stop_counter, lr_reduce_counter = 0, 0

    for epoch in tqdm(range(Config.num_epochs)):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for images, ecg_signals, clinical, labels in tqdm(train_loader):
            images, ecg_signals, clinical, labels = (
                images.to(Config.device),
                ecg_signals.to(Config.device),
                clinical.to(Config.device),
                labels.to(Config.device)
            )
            optimizer.zero_grad()
            outputs = model(images, ecg_signals, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # === Validation ===
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, ecg_signals, clinical, labels in tqdm(val_loader):
                images, ecg_signals, clinical, labels = (
                    images.to(Config.device),
                    ecg_signals.to(Config.device),
                    clinical.to(Config.device),
                    labels.to(Config.device)
                )
                outputs = model(images, ecg_signals, clinical)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        if avg_val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_inner.pth"))
            print(f"✅ At epoch {epoch+1} Saved best model to best_inner.pth")
            min_val_loss = avg_val_loss
            early_stop_counter, lr_reduce_counter = 0, 0
        else:
            early_stop_counter += 1
            lr_reduce_counter += 1
            if lr_reduce_counter >= 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                print(f"🔻 LR reduced")
                lr_reduce_counter = 0
            if early_stop_counter >= Config.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    writer.flush()
    writer.close()

def test_outer(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, ecg_signals, clinical, labels in test_loader:
            images, ecg_signals, clinical = (
                images.to(device),
                ecg_signals.to(device),
                clinical.to(device)
            )
            outputs = model(images, ecg_signals, clinical)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return auc

def main():
    labels_df, ecg_signals, clinical_df, labels, indices, transform = get_all_data(Config)
    outer_skf = StratifiedKFold(n_splits=Config.k_outer, shuffle=True, random_state=Config.seed)
    outer_aucs = []

    modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_skf.split(indices, labels)):
        print(f"\n=== Outer Fold {outer_fold+1}/{Config.k_outer} ===")
        
        fold_dir = os.path.join(checkpoint_dir, f"outer_fold_{outer_fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # === Inner CV ===
        inner_skf = StratifiedKFold(n_splits=Config.k_inner, shuffle=True, random_state=Config.seed)
        for inner_fold, (inner_train_idx, inner_val_idx) in tqdm(enumerate(inner_skf.split(train_val_idx, labels[train_val_idx]))):
            print(f"--- Inner Fold {inner_fold+1}/{Config.k_inner} ---")
            train_inner(train_val_idx[inner_train_idx], train_val_idx[inner_val_idx],
                        labels_df, ecg_signals, clinical_df, transform, fold_dir)

        # === Outer Test ===
        test_indices = labels_df.iloc[test_idx]['index'].tolist()
        train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_val_idx)]
        ecg_scaler = StandardScaler().fit(train_ecg)
        train_clinical = clinical_df[clinical_df['index'].isin(train_val_idx)].drop(columns=['index'])
        clinical_scaler = StandardScaler().fit(train_clinical)

        test_ds = ECGMultimodalDataset(test_indices, labels_df, ecg_signals, clinical_df,
                                       ecg_scaler, clinical_scaler, transform, config=Config)
        test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2)

        model = ECGMultimodalModel(Config).to(Config.device)
        model.load_state_dict(torch.load(os.path.join(fold_dir, "best_inner.pth"), weights_only=True))

        auc = test_outer(model, test_loader, Config.device)
        print(f"✅ Outer Fold {outer_fold+1} AUC: {auc:.4f}")
        outer_aucs.append(auc)

    print("\n=== Final Nested CV Results ===")
    for i, auc in enumerate(outer_aucs):
        print(f"Fold {i+1}: AUC = {auc:.4f}")
    print(f"Mean AUC: {np.mean(outer_aucs):.4f}")

if __name__ == "__main__":
    main()
