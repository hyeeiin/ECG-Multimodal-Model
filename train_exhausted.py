# train.py

import os
import time
import itertools
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


def train_inner(train_idx, val_idx, labels_df, ecg_signals, clinical_df, transform, run_dir):
    train_indices = labels_df.iloc[train_idx]['index'].tolist()
    val_indices = labels_df.iloc[val_idx]['index'].tolist()

    train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_indices)]
    ecg_scaler = StandardScaler().fit(train_ecg)
    train_clinical = clinical_df[clinical_df['index'].isin(train_indices)].drop(columns=['index'])
    clinical_scaler = StandardScaler().fit(train_clinical)

    train_ds = ECGMultimodalDataset(train_indices, labels_df, ecg_signals, clinical_df,
                                    ecg_scaler, clinical_scaler, transform, config=Config)
    val_ds = ECGMultimodalDataset(val_indices, labels_df, ecg_signals, clinical_df,
                                  ecg_scaler, clinical_scaler, transform, config=Config)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    model = ECGMultimodalModel(Config).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.lr)

    writer = SummaryWriter(os.path.join(run_dir, "inner_logs"))
    min_val_loss = float('inf')
    early_stop_counter, lr_reduce_counter = 0, 0

    for epoch in tqdm(range(Config.num_epochs)):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for images, ecg_signals, clinical, labels in train_loader:
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

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, ecg_signals, clinical, labels in val_loader:
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
            torch.save(model.state_dict(), os.path.join(run_dir, "best_inner.pth"))
            min_val_loss = avg_val_loss
            early_stop_counter, lr_reduce_counter = 0, 0
        else:
            early_stop_counter += 1
            lr_reduce_counter += 1
            if lr_reduce_counter >= 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                lr_reduce_counter = 0
            if early_stop_counter >= Config.patience:
                break

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
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=Config.seed)
    fold_indices = list(skf.split(indices, labels))

    all_aucs = []

    modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_count = 0

    for test_fold in range(10):
        for val_fold in range(10):
            if val_fold == test_fold:
                continue

            run_count += 1
            run_dir = os.path.join(checkpoint_dir, f"Run_{run_count}_test{test_fold}_val{val_fold}")
            os.makedirs(run_dir, exist_ok=True)

            test_idx = fold_indices[test_fold][1]
            val_idx = fold_indices[val_fold][1]
            train_idx = np.concatenate([
                fold_indices[i][1] for i in range(10) if i != test_fold and i != val_fold
            ])

            # === Inner CV ===
            train_inner(train_idx, val_idx, labels_df, ecg_signals, clinical_df, transform, run_dir)

            # === Outer Test ===
            train_ecg = ecg_signals.loc[ecg_signals.index.isin(train_idx)]
            ecg_scaler = StandardScaler().fit(train_ecg)
            train_clinical = clinical_df[clinical_df['index'].isin(train_idx)].drop(columns=['index'])
            clinical_scaler = StandardScaler().fit(train_clinical)

            test_indices = labels_df.iloc[test_idx]['index'].tolist()

            test_ds = ECGMultimodalDataset(test_indices, labels_df, ecg_signals, clinical_df,
                                           ecg_scaler, clinical_scaler, transform, config=Config)
            test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2)

            model = ECGMultimodalModel(Config).to(Config.device)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best_inner.pth"), weights_only=True))

            auc = test_outer(model, test_loader, Config.device)
            print(f"✅ Run {run_count}: Test Fold {test_fold} Val Fold {val_fold} → AUC: {auc:.4f}")
            all_aucs.append(auc)

    print("\n=== Final Exhaustive CV Results ===")
    print(f"Total Runs: {len(all_aucs)}")
    print(f"Mean AUC: {np.mean(all_aucs):.4f}")


if __name__ == "__main__":
    main()
