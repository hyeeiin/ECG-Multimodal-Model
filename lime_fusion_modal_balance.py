# lime_fusion_modal_balance.py

import lime
import lime.lime_tabular
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from config import Config
from dataset import get_dataloaders
from multimodal_paper_modal_balance import ECGMultimodalModel

class FusionClassifierWrapper(torch.nn.Module):
    def __init__(self, fusion_classifier):
        super().__init__()
        self.fusion_classifier = fusion_classifier

    def forward(self, fusion_embedding):
        return self.fusion_classifier(fusion_embedding)

# def get_embedding_batches(model, loader, device, max_samples=100):
#     model.eval()
#     embeddings = []
#     normal_samples = 0
#     abnormal_samples = 0

#     with torch.no_grad():
#         for images, ecg_signals, clinical, labels in tqdm(loader, desc="Extract BG Embeddings"):
#             images, ecg_signals, clinical = (
#                 images.to(device), ecg_signals.to(device), clinical.to(device)
#             )

#             img_feat = model.image_encoder(images)
#             img_feat = model.image_norm(img_feat)

#             signal_feat = model.signal_encoder(ecg_signals.unsqueeze(1))
#             signal_feat = model.signal_norm(signal_feat)

#             clinical_feat = model.clinical_encoder(clinical)
#             clinical_feat = model.clinical_norm(clinical_feat)

#             fused, soft_weights = model.attention_fusion(img_feat, signal_feat, clinical_feat)
#             embeddings.append(fused.cpu())

#             normal_samples += (labels == 0).sum().item()
#             abnormal_samples += (labels == 1).sum().item()

#             if len(embeddings) * fused.size(0) >= max_samples:
#                 break

#     bg_tensor = torch.cat(embeddings, dim=0)
#     print(f"✅ BG shape: {bg_tensor.shape} , Normal samples: {normal_samples}, Abnormal samples: {abnormal_samples}")
#     return bg_tensor.numpy()  # LIME은 numpy array 필요

def get_embedding_batches(model, loader, device, max_samples_per_class=50):
    model.eval()
    embeddings = []
    normal_samples = 0
    abnormal_samples = 0

    with torch.no_grad():
        for images, ecg_signals, clinical, labels in tqdm(loader, desc="Extract BG Embeddings"):
            if normal_samples < max_samples_per_class or abnormal_samples < max_samples_per_class:
                images, ecg_signals, clinical = (
                    images.to(device), ecg_signals.to(device), clinical.to(device)
                )

                img_feat = model.image_encoder(images)
                img_feat = model.image_norm(img_feat)

                signal_feat = model.signal_encoder(ecg_signals.unsqueeze(1))
                signal_feat = model.signal_norm(signal_feat)

                clinical_feat = model.clinical_encoder(clinical)
                clinical_feat = model.clinical_norm(clinical_feat)

                fused, _ = model.attention_fusion(img_feat, signal_feat, clinical_feat)
                embeddings.append(fused.cpu())

                # 레이블에 따라 카운트 증가
                for label in labels:
                    if label.item() == 0 and normal_samples < max_samples_per_class:
                        normal_samples += 1
                    elif label.item() == 1 and abnormal_samples < max_samples_per_class:
                        abnormal_samples += 1

            # 두 클래스 모두 50개에 도달하면 중단
            if normal_samples >= max_samples_per_class and abnormal_samples >= max_samples_per_class:
                break

    bg_tensor = torch.cat(embeddings, dim=0)
    print(f"✅ BG shape: {bg_tensor.shape}, Normal samples: {normal_samples}, Abnormal samples: {abnormal_samples}")
    return bg_tensor.numpy() 

def main(MODEL_PATH):
    device = torch.device(Config.device)
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    model = ECGMultimodalModel(Config).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/{MODEL_PATH}.pth', map_location=device, weights_only=True))
    model.eval()

    fusion_model = FusionClassifierWrapper(model.fusion_classifier).to(device)

    # === Background data for LIME ===
    # bg_embeddings = get_embedding_batches(model, train_loader, device, max_samples=100)
    bg_embeddings = get_embedding_batches(model, train_loader, device, max_samples_per_class=50)

    # === LIME Explainer ===
    # Fusion input은 [B, 512+128+32] 차원, continuous feature로 처리
    feature_names = (
        [f"img_{i}" for i in range(256)] +
        [f"sig_{i}" for i in range(256)] +
        [f"clin_{i}" for i in range(256)]
    )
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg_embeddings,
        feature_names=feature_names,
        class_names=['Normal', 'Abnormal'],
        mode='classification'
    )

    # === Wrapper for LIME prediction ===
    def predict_fn(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = fusion_model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    results = []

    for images, ecg_signals, clinical, labels in tqdm(test_loader, desc="LIME Test"):
        images, ecg_signals, clinical = (
            images.to(device), ecg_signals.to(device), clinical.to(device)
        )

        img_feat = model.image_encoder(images)
        img_feat = model.image_norm(img_feat)

        signal_feat = model.signal_encoder(ecg_signals.unsqueeze(1))
        signal_feat = model.signal_norm(signal_feat)

        clinical_feat = model.clinical_encoder(clinical)
        clinical_feat = model.clinical_norm(clinical_feat)

        fused, soft_weights = model.attention_fusion(img_feat, signal_feat, clinical_feat)
        fused = fused.detach().cpu().numpy()  # LIME은 numpy array 필요

        n_img = img_feat.shape[1]
        n_signal = signal_feat.shape[1]
        n_clinical = clinical_feat.shape[1]

        for b in range(fused.shape[0]):
            # LIME 설명 생성
            explanation = explainer.explain_instance(
                fused[b], predict_fn, num_features=len(feature_names), num_samples=1000
            )
            lime_weights = {f: w for f, w in explanation.local_exp[1]}  # Class 1 (Abnormal)에 대한 가중치

            # Modality별 기여도 계산
            img_contrib = np.sum([np.abs(lime_weights.get(i, 0)) for i in range(n_img)])
            sig_contrib = np.sum([np.abs(lime_weights.get(i, 0)) for i in range(n_img, n_img+n_signal)])
            clin_contrib = np.sum([np.abs(lime_weights.get(i, 0)) for i in range(n_img+n_signal, n_img+n_signal+n_clinical)])

            total = img_contrib + sig_contrib + clin_contrib

            results.append({
                'Sample_ID': len(results) + 1,
                'Image_%': img_contrib / total * 100 if total > 0 else 0,
                'Signal_%': sig_contrib / total * 100 if total > 0 else 0,
                'Clinical_%': clin_contrib / total * 100 if total > 0 else 0,
                'Label': labels[b].item()
            })

    df = pd.DataFrame(results)
    modelname = MODEL_PATH.split('/')[0]
    df.to_csv(f'./output/lime/{modelname}_lime_fusion_attention.csv', index=False)
    print(df.head())

    # === Attention weight 확인 ===
    attn_weights = soft_weights.detach().cpu()
    print(f"🔑 Attention Weights (Softmax): "
          f"Image={attn_weights[0].item():.4f} | "
          f"Signal={attn_weights[1].item():.4f} | "
          f"Clinical={attn_weights[2].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME ANALYSIS")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file (e.g., 0619_113448/epoch1)")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    main(MODEL_PATH)