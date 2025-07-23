# shap_fusion_paper.py

import shap
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from multimodal_paper import ECGMultimodalModel
import argparse

class FusionClassifierWrapper(torch.nn.Module):
    """
    Fusion classifier만 따로 SHAP 대상 래퍼.
    """
    def __init__(self, fusion_classifier):
        super(FusionClassifierWrapper, self).__init__()
        self.fusion_classifier = fusion_classifier

    def forward(self, fusion_embedding):
        return self.fusion_classifier(fusion_embedding)


def get_embedding_batches(model, loader, device, max_samples=100):
    """
    background embedding 추출 (train_loader에서 일부)
    """
    model.eval()
    embeddings = []

    with torch.no_grad():  # background는 grad 필요 없음
        for images, ecg_signals, clinical, labels in tqdm(loader, desc="Extract BG Embeddings"):
            images, ecg_signals, clinical = (
                images.to(device), ecg_signals.to(device), clinical.to(device)
            )

            img_feat = model.image_encoder(images)
            ecg_signals = ecg_signals.unsqueeze(1)  # [B, L] → [B, 1, L]
            signal_feat = model.signal_encoder(ecg_signals)
            clinical_feat = model.clinical_encoder(clinical)

            fusion_input = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
            embeddings.append(fusion_input.cpu())

            if len(embeddings) * fusion_input.size(0) >= max_samples:
                break

    bg_tensor = torch.cat(embeddings, dim=0)
    print(f"✅ BG embedding shape: {bg_tensor.shape}")
    return bg_tensor


def main(MODEL_PATH):
    device = torch.device(Config.device)

    train_loader, val_loader, test_loader = get_dataloaders(Config)

    model = ECGMultimodalModel(Config).to(device)
    modeltime = MODEL_PATH
    model.load_state_dict(torch.load(f'./checkpoints/{modeltime}.pth', map_location=device, weights_only=True))
    model.eval()

    fusion_model = FusionClassifierWrapper(model.fusion_classifier).to(device)

    # === BG embedding ===
    bg_embeddings = get_embedding_batches(model, train_loader, device, max_samples=100).to(device)
    bg_embeddings.requires_grad_()

    explainer = shap.DeepExplainer(fusion_model, bg_embeddings)

    results = []

    for images, ecg_signals, clinical, labels in tqdm(test_loader, desc="SHAP Test"):
        images = images.to(device)
        ecg_signals = ecg_signals.to(device)
        ecg_signals = ecg_signals.unsqueeze(1)  # [B, L] → [B, 1, L]
        clinical = clinical.to(device)

        # ✅ grad 추적 필요! no torch.no_grad!
        img_feat = model.image_encoder(images)
        signal_feat = model.signal_encoder(ecg_signals)
        clinical_feat = model.clinical_encoder(clinical)

        fusion_input = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)
        fusion_input.requires_grad_()

        shap_values = explainer.shap_values(fusion_input)
        # shap_vals = shap_values[0].detach().cpu().numpy()  # [B, total_dim]
        shap_vals = shap_values[0]

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

    df = pd.DataFrame(results)
    modelname = modeltime.split('/')[0]
    df.to_csv(f'./output/shap/{modelname}_shap_fusion.csv', index=False)
    print(f"✅ Saved: ./output/shap/{modelname}_shap_fusion.csv")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP ANALYSIS")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file (e.g., 0619_113448/epoch1)")

    args = parser.parse_args()
    MODEL_PATH = args.model_path
    main(MODEL_PATH)
