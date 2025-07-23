# shap_fusion_paper_modal_balance.py

import shap
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

#             if len(embeddings) * fused.size(0) >= max_samples:
#                 break

#     bg_tensor = torch.cat(embeddings, dim=0)
#     print(f"✅ BG shape: {bg_tensor.shape}")
#     return bg_tensor

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
                # print(f"Image feat shape: {img_feat.shape}, mean: {img_feat.mean().item()}, std: {img_feat.std().item()}")

                signal_feat = model.signal_encoder(ecg_signals.unsqueeze(1))
                signal_feat = model.signal_norm(signal_feat)
                # print(f"Signal feat shape: {signal_feat.shape}, mean: {signal_feat.mean().item()}, std: {signal_feat.std().item()}")

                clinical_feat = model.clinical_encoder(clinical)
                clinical_feat = model.clinical_norm(clinical_feat)
                # print(f"Clinical feat shape: {clinical_feat.shape}, mean: {clinical_feat.mean().item()}, std: {clinical_feat.std().item()}")

                fused, _ = model.attention_fusion(img_feat, signal_feat, clinical_feat)
                embeddings.append(fused.cpu())

                for label in labels:
                    if label.item() == 0 and normal_samples < max_samples_per_class:
                        normal_samples += 1
                    elif label.item() == 1 and abnormal_samples < max_samples_per_class:
                        abnormal_samples += 1

            if normal_samples >= max_samples_per_class and abnormal_samples >= max_samples_per_class:
                break

    bg_tensor = torch.cat(embeddings, dim=0)
    print(f"✅ BG shape: {bg_tensor.shape}, Normal samples: {normal_samples}, Abnormal samples: {abnormal_samples}")
    print(f"BG tensor stats - mean: {bg_tensor.mean().item()}, std: {bg_tensor.std().item()}")
    return bg_tensor


def main(MODEL_PATH):
    device = torch.device(Config.device)
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    model = ECGMultimodalModel(Config).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/{MODEL_PATH}.pth', map_location=device, weights_only=True))
    model.eval()


    fusion_fc = model.fusion_classifier[0].weight.detach().cpu().numpy()  # shape: [128, 768] 예시
    # print(f"Fusion FC weight shape: {fusion_fc.shape}")

    modal_dim = model.modal_dim  # 예: 256
    image_dim = model.modal_dim
    signal_dim = model.modal_dim
    clinical_dim = model.modal_dim
    img_chunk = fusion_fc[:, :image_dim]
    signal_chunk = fusion_fc[:, image_dim:image_dim + signal_dim]
    clinical_chunk = fusion_fc[:, image_dim + signal_dim:]

    img_norm = np.linalg.norm(img_chunk)
    sig_norm = np.linalg.norm(signal_chunk)
    clin_norm = np.linalg.norm(clinical_chunk)

    print(f"🧩 Fusion FC weight chunk norms:")
    print(f"   Image chunk norm   : {img_norm:.4f}")
    print(f"   Signal chunk norm  : {sig_norm:.4f}")
    print(f"   Clinical chunk norm: {clin_norm:.4f}")


    fusion_model = FusionClassifierWrapper(model.fusion_classifier).to(device)

    # === BG embedding
    # bg_embeddings = get_embedding_batches(model, train_loader, device, max_samples=100).to(device)
    bg_embeddings = get_embedding_batches(model, train_loader, device, max_samples_per_class=50).to(device)
    bg_embeddings.requires_grad_()
    # print(f"BG embeddings requires_grad: {bg_embeddings.requires_grad}")

    # explainer = shap.DeepExplainer(fusion_model, bg_embeddings)
    explainer = shap.GradientExplainer(fusion_model, bg_embeddings)

    results = []

    for images, ecg_signals, clinical, labels in tqdm(test_loader, desc="SHAP Test"):
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
        fused.requires_grad_()
        print(f"Fused shape: {fused.shape}, soft_weights: {soft_weights}") # torch.Size([16, 768])

        fused = torch.cat([img_feat, signal_feat, clinical_feat], dim=1)

        shap_values = explainer.shap_values(fused)
        print(f"Shap_values type: {type(shap_values)}, shape: {shap_values.shape}") # shape: (16, 768, 2)
        # print(f"Shap_values min: {shap_values.min()}, max: {shap_values.max()}, mean: {shap_values.mean()}")
        # abs_sum = np.abs(shap_values).sum()
        # print(f"Absolute sum of shap_values: {abs_sum}")
        # shap_vals = shap_values[0]
        # print(f"Shap_vals shape for class 0: {shap_vals.shape}") # (768, 2)

        # fused.retain_grad()
        # output = fusion_model(fused)
        # output.sum().backward()
        # print(f"Fused grad shape: {fused.grad.shape}, grad sum per modal: {fused.grad[:, :256].sum()}, {fused.grad[:, 256:512].sum()}, {fused.grad[:, 512:].sum()}")

        n_img = img_feat.shape[1]
        n_signal = signal_feat.shape[1]
        n_clinical = clinical_feat.shape[1]
        print(f"n_img : {n_img}, n_signal : {n_signal}, n_clinical : {n_clinical}")

        for b in range(shap_values.shape[0]):  # 16 샘플
            for class_idx in range(shap_values.shape[2]):  # 2 클래스
                shap_vals_b = shap_values[b, :, class_idx]  # [768] 벡터
                sample_vals = np.abs(shap_vals_b)
                # print(f"Sample {b+1}, Class {class_idx} - sample_vals shape: {sample_vals.shape}, first 10 values: {sample_vals[:10]}")

                # img_contrib = np.sum(sample_vals[:n_img])
                # sig_contrib = np.sum(sample_vals[n_img:n_img+n_signal])
                # clin_contrib = np.sum(sample_vals[n_img+n_signal:])
                # total = img_contrib + sig_contrib + clin_contrib
                img_contrib = np.mean(sample_vals[:n_img])
                sig_contrib = np.mean(sample_vals[n_img:n_img+n_signal])
                clin_contrib = np.mean(sample_vals[n_img+n_signal:])
                total = img_contrib + sig_contrib + clin_contrib
                # print(f"img: {img_contrib}, sig: {sig_contrib}, cli: {clin_contrib}, total: {total}")

                results.append({
                    'Sample_ID': (len(results))//2 + 1,
                    'Image_%': img_contrib / total * 100,
                    'Signal_%': sig_contrib / total * 100,
                    'Clinical_%': clin_contrib / total * 100,
                    'Label': labels[b].item(),
                    'Class': class_idx
                })

    df = pd.DataFrame(results)
    modelname = MODEL_PATH.split('/')[0]
    df.to_csv(f'./output/shap/{modelname}_shap_fusion.csv', index=False)
    print(df.head())

    # === Attention weight 확인 ===
    # attn = model.attention_fusion
    # print(f"🔑 Learned Attention Weights → "
    #       f"Image: {attn.image_w.item():.4f} | "
    #       f"Signal: {attn.signal_w.item():.4f} | "
    #       f"Clinical: {attn.clinical_w.item():.4f}")
    attn_weights = soft_weights.detach().cpu()
    print(f"🔑 Attention Weights (Softmax): "
              f"Image={attn_weights[0].item():.4f} | "
              f"Signal={attn_weights[1].item():.4f} | "
              f"Clinical={attn_weights[2].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP ANALYSIS")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file (e.g., 0619_113448/epoch1)")

    args = parser.parse_args()
    MODEL_PATH = args.model_path
    main(MODEL_PATH)
