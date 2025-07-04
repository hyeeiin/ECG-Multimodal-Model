# train.py

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from multimodal import ECGMultimodalModel  # 모델 클래스명 맞춰주세요!

def main():
    # Seed 고정
    torch.manual_seed(Config.seed)

    # Device
    device = torch.device(Config.device)
    print(f"📍 Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Model
    model = ECGMultimodalModel(Config).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.lr)

    # TensorBoard & Checkpoint
    modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
    writer = SummaryWriter(f"runs/{modeltime}")

    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping vars
    min_val_loss = float('inf')
    early_stop_counter = 0

    # Epoch loop
    for epoch in range(Config.num_epochs):
        print(f"\n=== Epoch [{epoch+1}/{Config.num_epochs}] ===")

        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        loop = tqdm(train_loader, desc="Training")

        for images, ecg_signals, clinical, labels in loop:
            images, ecg_signals, clinical, labels = (
                images.to(device), ecg_signals.to(device), clinical.to(device), labels.to(device)
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

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            loop = tqdm(val_loader, desc="Validating")
            for images, ecg_signals, clinical, labels in loop:
                images, ecg_signals, clinical, labels = (
                    images.to(device), ecg_signals.to(device), clinical.to(device), labels.to(device)
                )

                outputs = model(images, ecg_signals, clinical)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # TensorBoard logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # Save best checkpoint
        if avg_val_loss < min_val_loss:
            ckpt_path = os.path.join(checkpoint_dir, f"best_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved best model to {ckpt_path}")
            min_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} (patience={Config.patience})")
                break

    writer.close()
    print("🎉 Training completed!")

    # Optional: test set evaluation
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, ecg_signals, clinical, labels in tqdm(test_loader, desc="Testing"):
            images, ecg_signals, clinical, labels = (
                images.to(device), ecg_signals.to(device), clinical.to(device), labels.to(device)
            )
            outputs = model(images, ecg_signals, clinical)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_acc = correct_test / total_test
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
