import shap
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from multimodal import ECGMultimodalModel
from fusion_classifier import FusionClassifierWrapper


def get_embedding_batches(model, loader, device, max_samples=100):
    """
    Train loader에서 일부 샘플로 background embedding 추출 (no_grad ok!)
    """
    model.eval()
    embeddings = []

    with torch.no_grad():  # ✅ background는 grad 불필요!
        for images, ecg_signals, clinical, labels in tqdm(loader, desc="Extract Embeddings"):
            images = images.to(device)
            ecg_signals = ecg_signals.to(device)
            ecg_signals = ecg_signals.unsqueeze(1)
            clinical = clinical.to(device)

            img_feat = model.image_encoder(images)
            signal_feat = model.signal_encoder(ecg_signals)
            clinical_feat = model.clinical_encoder(clinical)

            fusion_input = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
            embeddings.append(fusion_input.cpu())  # tensor 유지!

            if len(embeddings) * fusion_input.size(0) >= max_samples:
                break

    background_tensor = torch.cat(embeddings, dim=0)
    print(f"✅ Background embedding shape: {background_tensor.shape}")
    return background_tensor


def main():
    device = torch.device(Config.device)

    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # === 모델 로드 ===
    model = ECGMultimodalModel(Config).to(device)
    modeltime = '0709_143114/best_epoch1'
    ckpt = torch.load(f'./checkpoints/{modeltime}.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    # Fusion classifier만 래핑
    fusion_model = FusionClassifierWrapper(model.classifier).to(device)

    # === 배경 embedding ===
    background_embeddings = get_embedding_batches(model, train_loader, device, max_samples=100).to(device)
    background_embeddings.requires_grad_()  # ✅ gradient 필요!

    # === SHAP explainer 준비 ===
    explainer = shap.DeepExplainer(fusion_model, background_embeddings)

    results = []

    # === test_loader 돌면서 샘플별 SHAP ===
    for images, ecg_signals, clinical, labels in tqdm(test_loader, desc="SHAP Inference"):
        images = images.to(device)
        ecg_signals = ecg_signals.to(device)
        ecg_signals = ecg_signals.unsqueeze(1)
        clinical = clinical.to(device)

        # ✅ forward시 grad 켜짐 (no torch.no_grad)
        img_feat = model.image_encoder(images)
        signal_feat = model.signal_encoder(ecg_signals)
        clinical_feat = model.clinical_encoder(clinical)

        fusion_input = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
        fusion_input.requires_grad_()  # ✅ gradient 추적 보장!

        # === SHAP 계산
        shap_values = explainer.shap_values(fusion_input)
        # shap_vals = shap_values[0].detach().cpu().numpy()  # [B, total_dim]
        shap_vals = shap_values[0]  # [B, total_dim]

        n_img = img_feat.shape[1]
        n_signal = signal_feat.shape[1]
        n_clinical = clinical_feat.shape[1]

        for b in range(fusion_input.size(0)):
            sample_vals = np.abs(shap_vals[b])
            img_contrib = np.sum(sample_vals[:n_img])
            sig_contrib = np.sum(sample_vals[n_img:n_img+n_signal])
            clin_contrib = np.sum(sample_vals[n_img+n_signal:])

            total = img_contrib + sig_contrib + clin_contrib

            results.append({
                'Sample_ID': len(results) + 1,
                'Image_%': img_contrib / total * 100,
                'Signal_%': sig_contrib / total * 100,
                'Clinical_%': clin_contrib / total * 100,
                'Label': labels[b].item()
            })

    # === CSV 저장 ===
    df = pd.DataFrame(results)
    modelname = modeltime.split('/')[0]
    df.to_csv(f'./output/shap/{modelname}_fusionXAI.csv', index=False)
    print("✅ CSV 저장 완료!")
    print(df.head())


if __name__ == "__main__":
    main()
