# train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from FinalProject.config2 import config
from FinalProject.dataset2 import get_dataloaders
from FinalProject.multimodal2 import MultiModalNet
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import time

# Set random seed
torch.manual_seed(config.SEED)

# Load data
clinical_df = pd.read_csv(config.CLINICAL_CSV)
ecg_df = pd.read_csv(config.ECG_CSV)
labels = clinical_df["label"].values
image_paths = [os.path.join(config.IMAGE_DIR, f"{id}.jpg") for id in clinical_df["id"].values]

# Dataloaders
train_loader, val_loader = get_dataloaders(config, clinical_df.drop(columns="label"), ecg_df, image_paths, labels)

# Model
model = MultiModalNet(config.CLINICAL_INPUT_DIM, config.ECG_INPUT_DIM, config.NUM_CLASSES, config.DROPOUT, config.PRETRAINED)
device = config.DEVICE
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

# checkpoint dir
modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
checkpoint_dir = os.path.join('checkpoints', modeltime)
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard Writer
writer = SummaryWriter(f"runs/{modeltime}")

# Checkpoint directory
checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

min_valid_loss = float('inf')
early_stop_counter = 0

for epoch in tqdm(range(config.EPOCHS)):
    model.train()
    train_loss = 0.0
    correct_train, total_train = 0, 0

    # For branch loss logging
    branch_train_losses = {"image": 0.0, "clinical": 0.0, "ecg": 0.0, "fusion": 0.0}

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

    for img, clinical, ecg, label in loop:
        img, clinical, ecg, label = img.to(device), clinical.to(device), ecg.to(device), label.to(device)
        
        optimizer.zero_grad()
        img_out, clinical_out, ecg_out, fusion_out = model(img, clinical, ecg)

        # Individual losses
        loss_img = criterion(img_out, label)
        loss_clinical = criterion(clinical_out, label)
        loss_ecg = criterion(ecg_out, label)
        loss_fusion = criterion(fusion_out, label)

        # Weighted sum of losses
        total_loss = (
            config.LAMBDA_IMAGE * loss_img +
            config.LAMBDA_CLINICAL * loss_clinical +
            config.LAMBDA_ECG * loss_ecg +
            config.LAMBDA_FUSION * loss_fusion
        )
        
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        branch_train_losses["image"] += loss_img.item()
        branch_train_losses["clinical"] += loss_clinical.item()
        branch_train_losses["ecg"] += loss_ecg.item()
        branch_train_losses["fusion"] += loss_fusion.item()

        _, predicted = torch.max(fusion_out, 1)
        total_train += label.size(0)
        correct_train += predicted.eq(label).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0
    branch_val_losses = {"image": 0.0, "clinical": 0.0, "ecg": 0.0, "fusion": 0.0}

    with torch.no_grad():
        for img, clinical, ecg, label in val_loader:
            img, clinical, ecg, label = img.to(device), clinical.to(device), ecg.to(device), label.to(device)
            img_out, clinical_out, ecg_out, fusion_out = model(img, clinical, ecg)
            
            loss_img = criterion(img_out, label)
            loss_clinical = criterion(clinical_out, label)
            loss_ecg = criterion(ecg_out, label)
            loss_fusion = criterion(fusion_out, label)

            total_val_loss = (
                config.LAMBDA_IMAGE * loss_img +
                config.LAMBDA_CLINICAL * loss_clinical +
                config.LAMBDA_ECG * loss_ecg +
                config.LAMBDA_FUSION * loss_fusion
            )
            
            val_loss += total_val_loss.item()
            branch_val_losses["image"] += loss_img.item()
            branch_val_losses["clinical"] += loss_clinical.item()
            branch_val_losses["ecg"] += loss_ecg.item()
            branch_val_losses["fusion"] += loss_fusion.item()

            _, predicted = torch.max(fusion_out, 1)
            total_val += label.size(0)
            correct_val += predicted.eq(label).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val

    # TensorBoard Logging
    writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation_Total", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    # Branch Loss Logging
    for branch in branch_train_losses:
        writer.add_scalar(f"Loss/Train_{branch}", branch_train_losses[branch]/len(train_loader), epoch)
        writer.add_scalar(f"Loss/Validation_{branch}", branch_val_losses[branch]/len(val_loader), epoch)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save checkpoint
    if avg_val_loss < min_valid_loss:
        torch.save(model.state_dict(), f'checkpoints/{modeltime}/epoch{epoch}.pth')
        min_valid_loss = avg_val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {min_valid_loss:.4f}")
            break

writer.flush()
writer.close()
print("✅ Training finished!")
