# train_final.py

import pandas as pd
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from multimodal import ECGMultimodalModel

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    roc_curve
import matplotlib.pyplot as plt
import numpy as np


def main():
    torch.manual_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"📍 Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(Config)

    model = ECGMultimodalModel(Config).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=Config.lr)

    # ✅ Encoder freeze (fusion classifier만 학습)
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.signal_encoder.parameters():
        param.requires_grad = False
    for param in model.clinical_encoder.parameters():
        param.requires_grad = False

    # ✅ Optimizer 재설정: 학습 가능한 파라미터만 최적화
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr)

    modeltime = time.strftime('%m%d_%H%M%S', time.localtime())
    writer = SummaryWriter(f"runs/{modeltime}")

    checkpoint_dir = os.path.join(Config.checkpoint_dir, modeltime)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = float('inf')
    early_stop_counter = 0
    lr_reduce_counter = 0

    for epoch in tqdm(range(Config.num_epochs)):
        print(f"\n=== Epoch [{epoch + 1}/{Config.num_epochs}] ===")
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, ecg_signals, clinical, labels in tqdm(train_loader, desc="Training"):
            images, ecg_signals, clinical, labels = (
                images.to(device), ecg_signals.to(device), clinical.to(device), labels.to(device)
            )

            optimizer.zero_grad()

            img_logits, signal_logits, clinical_logits, fusion_logits, var_loss, soft_weights = model(images, ecg_signals, clinical)

            loss_img = criterion(img_logits, labels)
            loss_signal = criterion(signal_logits, labels)
            loss_clinical = criterion(clinical_logits, labels)
            loss_fusion = criterion(fusion_logits, labels)

            # total_loss = (
            #     loss_fusion + 1.0 * (loss_img + loss_signal + loss_clinical)
            #     + 0.1 * var_loss  # variance regularization 비율 튜닝!
            # )
            total_loss = loss_fusion + 0.1 * var_loss
            # total_loss = loss_fusion
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = fusion_logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_var_loss = 0.0
        with torch.no_grad():
            for images, ecg_signals, clinical, labels in tqdm(val_loader, desc="Validating"):
                images, ecg_signals, clinical, labels = (
                    images.to(device), ecg_signals.to(device), clinical.to(device), labels.to(device)
                )
                img_logits, signal_logits, clinical_logits, fusion_logits, var_loss, soft_weights = model(images, ecg_signals, clinical)
                loss_img = criterion(img_logits, labels)
                loss_signal = criterion(signal_logits, labels)
                loss_clinical = criterion(clinical_logits, labels)
                loss_fusion = criterion(fusion_logits, labels)

                # total_batch_loss = loss_fusion + 1.0 * (loss_img + loss_signal + loss_clinical) + 0.1 * var_loss
                total_batch_loss = loss_fusion + 0.1 * var_loss
                val_loss += total_batch_loss.item()
                val_var_loss += var_loss.item()

                _, predicted = fusion_logits.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_var_loss = val_var_loss / len(val_loader)
        val_acc = correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # === TensorBoard Logging ===
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("VarLoss/Val", avg_val_var_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # 🔑 Attention weights Logging
        attn_weights = soft_weights.detach().cpu()
        writer.add_scalar("AttentionWeights/Image_w", attn_weights[0].item(), epoch)
        writer.add_scalar("AttentionWeights/Signal_w", attn_weights[1].item(), epoch)
        writer.add_scalar("AttentionWeights/Clinical_w", attn_weights[2].item(), epoch)

        # print(f"🔑 Attention Weights: Image={attn.image_w.item():.4f} | "
        #       f"Signal={attn.signal_w.item():.4f} | Clinical={attn.clinical_w.item():.4f}")
        print(f"🔑 Attention Weights (Softmax): "
              f"Image={attn_weights[0].item():.4f} | "
              f"Signal={attn_weights[1].item():.4f} | "
              f"Clinical={attn_weights[2].item():.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"last.pth"))

        # Early stop & LR scheduler
        if avg_val_loss < min_val_loss:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pth"))
            print(f"✅ Saved best model to {ckpt_path}")
            min_val_loss = avg_val_loss
            early_stop_counter = 0
            lr_reduce_counter = 0
        else:
            early_stop_counter += 1
            lr_reduce_counter += 1

            if lr_reduce_counter >= 2:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr / 10
                    param_group['lr'] = new_lr
                print(f"🔻 LR reduced from {old_lr:.2e} to {new_lr:.2e} due to no improvement for 2 epochs.")
                lr_reduce_counter = 0

            if early_stop_counter >= Config.patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break

    writer.flush()
    writer.close()
    print("🎉 Training completed!")

    # Test loop
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best.pth")))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    img_acc, signal_acc, clinical_acc = [], [], []

    with torch.no_grad():
        for images, ecg_signals, clinical, labels, index in tqdm(test_loader, desc="Testing"):
            images, ecg_signals, clinical = images.to(device), ecg_signals.to(device), clinical.to(device)
            labels = labels.to(device)
            #
            # print("image shape: ",images.shape)
            # single_image = images[0] #(1,3,244,244)
            # image_np = single_image.squeeze(0).cpu().numpy()  # (3, 224, 224)
            #
            # with pd.ExcelWriter("./check/image_channels_bk2.xlsx") as writer:
            #     for i in range(3):
            #         df = pd.DataFrame(image_np[i])  # shape (224, 224)
            #         df.to_excel(writer, sheet_name=f"channel{i}", index=False)
            #
            # print("ecg: ",ecg_signals)
            #
            # print(clinical)

            image_logits, signal_logits, clinical_logits, fusion_logits, _, _ = model(images, ecg_signals, clinical)

            # img_acc.extend((image_logits.max(1)[1] == labels).float())
            # signal_acc.extend((signal_logits.max(1)[1] == labels).float())
            # clinical_acc.extend((clinical_logits.max(1)[1] == labels).float())

            probs = torch.softmax(fusion_logits, dim=1)[:, 1]  # class 1 확률만
            _, predicted = fusion_logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # print(index)
            # print(all_preds, all_probs, all_labels)

    # 각 branch의 예측 성능 확인
    # print(f"Image Acc: {img_acc.mean().item():.4f}, Signal Acc: {signal_acc.mean().item():.4f}, Clinical Acc: {clinical_acc.mean().item():.4f}")
    print(all_preds, all_probs, all_labels)
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

    # Test loop (LAST)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"last.pth")))
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    img_acc, signal_acc, clinical_acc = [], [], []

    with torch.no_grad():
        for images, ecg_signals, clinical, labels, index in tqdm(test_loader, desc="Testing"):
            images, ecg_signals, clinical = images.to(device), ecg_signals.to(device), clinical.to(device)
            labels = labels.to(device)

            image_logits, signal_logits, clinical_logits, fusion_logits, _, _ = model(images, ecg_signals, clinical)

            # img_acc.extend((image_logits.max(1)[1] == labels).float())
            # signal_acc.extend((signal_logits.max(1)[1] == labels).float())
            # clinical_acc.extend((clinical_logits.max(1)[1] == labels).float())

            probs = torch.softmax(fusion_logits, dim=1)[:, 1]  # class 1 확률만
            _, predicted = fusion_logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # print(index)

    # 각 branch의 예측 성능 확인
    # print(f"Image Acc: {img_acc.mean().item():.4f}, Signal Acc: {signal_acc.mean().item():.4f}, Clinical Acc: {clinical_acc.mean().item():.4f}")

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


if __name__ == "__main__":
    main()
