# multimodal_paper_modal_balance.py

import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_tabnet.tab_network import TabNetNoEmbeddings
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

class AttentionFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))  # image, signal, clinical
        self.norm = nn.LayerNorm(sum(dims))

    def forward(self, img_feat, signal_feat, clinical_feat):
        soft_weights = torch.softmax(self.weights, dim=0)
        # soft_weights = torch.sigmoid(self.weights)
        fused = torch.cat([
            soft_weights[0] * img_feat,
            soft_weights[1] * signal_feat,
            soft_weights[2] * clinical_feat
        ], dim=1)
        fused = self.norm(fused)
        return fused, soft_weights

# ✅ Squeeze-and-Excitation (SE) 블록 정의
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# ✅ Basic Residual Block (1D)
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# ✅ ResNet1D with SE
class ResNet1D_SE(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, base_filters=64):
        super(ResNet1D_SE, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = BasicBlock1D(base_filters, base_filters)
        self.layer2 = BasicBlock1D(base_filters, base_filters * 2, stride=2)
        self.layer3 = BasicBlock1D(base_filters * 2, base_filters * 4, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return self.classifier(x)


class ClinicalTabNetEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, device=None):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=latent_dim,
            n_d=latent_dim,
            n_a=latent_dim,
            n_steps=3,
            gamma=1.5,
            n_independent=2,
            n_shared=2
        ).to(device)

    def _move_buffers_to_device(self):
        """
        ✅ TabNet 내부 버퍼들을 디바이스에 맞게 이동
        """
        for name, buf in self.tabnet.named_buffers():
            if buf.device != self.device:
                # print(f"⚠️ Buffer {name} was on {buf.device} → moving to {self.device}")
                buf.data = buf.data.to(self.device)

        # ✅ 특수 group_attention_matrix 별도로 확인
        if hasattr(self.tabnet.encoder, 'group_attention_matrix'):
            if self.tabnet.encoder.group_attention_matrix.device != self.device:
                self.tabnet.encoder.group_attention_matrix = self.tabnet.encoder.group_attention_matrix.to(self.device)
                # print(f"✔️ Moved group_attention_matrix to {self.device}")

    def forward(self, x):
        """
        ✅ 학습/추론용 안전 forward:
        - TabNet 공식 forward만 사용 → 내부에서 encoder, prior 등 다 관리
        - 반환: output (latent vector), M_loss (마스크 loss)
        """
        x = x.to(self.device)
        out, M_loss = self.tabnet(x)
        return out, M_loss

    def load_pretrained_partial(self, weight_path):
        """
        ✅ output layer 제외하고 encoder만 가져오기
        """
        print(f"🔄 Loading partial TabNet weights from {weight_path}")
        saved_state = torch.load(weight_path, map_location=self.device)  # map_location 추가!!
        filtered_state = {k: v for k, v in saved_state.items() if 'final_mapping' not in k}
        self.tabnet.load_state_dict(filtered_state, strict=False)

        print(f"✅ Loaded TabNet encoder weights (latent_dim adapted to {self.latent_dim})")
        # for key in saved_state.keys():
        #     print(key)
        # TabNet의 initial_bn.running_mean 텐서 확인
        if 'encoder.tabnet.initial_bn.running_mean' in saved_state:
            input_dim = saved_state['encoder.tabnet.initial_bn.running_mean'].shape[0]
            print(f"✅ This TabNet model was trained with input_dim = {input_dim}")
        else:
            print("❌ Couldn't find 'encoder.tabnet.initial_bn.running_mean' in state_dict keys.")
        self._move_buffers_to_device()

    def visualize_masks(self, X, feature_names=None, save_dir="./shap", base_filename="mask"):
        """
        ✅ step별 mask 시각화:
        - TabNet의 forward_masks() 사용 (공식 지원)
        """
        self.tabnet.eval()

        # numpy 또는 tensor 입력 지원
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            outputs = self.tabnet.forward_masks(X_tensor)
            # forward_masks()는 (output, M_explain, masks) 반환
            if len(outputs) == 3:
                output, M_explain, masks = outputs
                masks = [mask.detach().cpu().numpy() for mask in masks]
            else:
                output, M_explain = outputs
                masks = []

            print("DEBUG type(M_explain):", type(M_explain))


        os.makedirs(save_dir, exist_ok=True)

        if feature_names is None:
            feature_names = [f"var_{i}" for i in range(X_tensor.shape[1])]

        if isinstance(M_explain, dict):
            step_masks = []
            for step, mask_tensor in M_explain.items():
                mask_np = mask_tensor.detach().cpu().numpy()
                step_masks.append(mask_np)

                # Step별 저장
                plt.figure(figsize=(12, 1))
                sns.heatmap(np.mean(mask_np, axis=0).reshape(1, -1), cmap="viridis",
                            cbar=True, xticklabels=feature_names)
                plt.title(f"Step Mask M[{step + 1}] (mean over batch)")
                step_filename = os.path.join(save_dir, f"{base_filename}_M[{step + 1}].png")
                plt.savefig(step_filename, bbox_inches="tight")
                print(f"✅ Saved: {step_filename}")
                plt.close()

            # === Aggregate ===
            M_agg_mean = np.mean(step_masks, axis=0)
            plt.figure(figsize=(12, 1))
            sns.heatmap(np.mean(M_agg_mean, axis=0).reshape(1, -1), cmap="viridis",
                        cbar=True, xticklabels=feature_names)
            plt.title("Aggregate Mask M_agg (mean over batch)")
            agg_filename = os.path.join(save_dir, f"{base_filename}_M_agg.png")
            plt.savefig(agg_filename, bbox_inches="tight")
            print(f"✅ Saved: {agg_filename}")
            plt.close()

        else:
            # Tensor면 그대로 처리
            M_explain_tensor = M_explain
            M_agg = M_explain_tensor.detach().cpu().numpy()
            M_agg_mean = np.mean(M_agg, axis=0)
            plt.figure(figsize=(12, 1))
            sns.heatmap(M_agg_mean.reshape(1, -1), cmap="viridis",
                        cbar=True, xticklabels=feature_names)
            plt.title("Aggregate Mask M_agg (mean over batch)")
            agg_filename = os.path.join(save_dir, f"{base_filename}_M_agg.png")
            plt.savefig(agg_filename, bbox_inches="tight")
            print(f"✅ Saved: {agg_filename}")
            plt.close()

    def extract_clinical_features(self, clinical):
        clinical = clinical.to(self.config.device)
        z, _ = self.clinical_encoder(clinical)
        return z

    def visualize_clinical_masks(self, clinical, feature_names, save_dir, base_filename):
        clinical = clinical.to(self.config.device)
        return self.clinical_encoder.visualize_masks(clinical, feature_names, save_dir, base_filename)




# ✅ Squeeze-and-Excitation (SE) 블록 정의
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# ✅ Basic Residual Block (1D)
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# ✅ ResNet1D with SE
class ResNet1D_SE(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, base_filters=64):
        super(ResNet1D_SE, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = BasicBlock1D(base_filters, base_filters)
        self.layer2 = BasicBlock1D(base_filters, base_filters * 2, stride=2)
        self.layer3 = BasicBlock1D(base_filters * 2, base_filters * 4, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return self.classifier(x)

class ECGMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ✅ 동일 dimension으로 맞추기
        self.modal_dim = 256  # image, signal, clinical 다 맞춤
        self.image_dim = 512
        self.signal_dim = 128
        self.clinical_dim = 32

        # ✅ Image encoder (ResNet18) + LayerNorm
        # self.image_encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder = models.resnet18()
        # self.image_encoder.fc = nn.Identity()
        # self.image_norm = nn.LayerNorm(512)

        checkpoint = torch.load('./checkpoints/0711_154435/last.pth', map_location='cpu')
        self.image_encoder.load_state_dict(checkpoint, strict=False)
        # self.load_pretrained_image_encoder(weight_path='./checkpoints/0711_154435/last.pth',
        #                         load_fc=False)

        # self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, self.modal_dim)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, self.image_dim)
        # self.image_norm = nn.LayerNorm(self.modal_dim)
        self.image_norm = nn.LayerNorm(self.image_dim)

        # ✅ Signal encoder (ResNet1D_SE) + LayerNorm
        self.signal_encoder = ResNet1D_SE(input_channels=1, num_classes=self.signal_dim)
        # self.signal_norm = nn.LayerNorm(128)

        # self.signal_encoder = SignalEncoder_1DTransformer_SE(
        #     input_channels=1,
        #     embedding_dim=128  # 기존 signal branch output dim과 맞춰!
        # )

        self.load_pretrained_signal_encoder(
            weight_path='./checkpoints/0716_172631/best.pth',
            load_fc=False  # True면 classifier까지, False면 feature extractor만
        )
        # self.signal_norm = nn.LayerNorm(self.modal_dim)
        self.signal_norm = nn.LayerNorm(self.signal_dim)

        # self.signal_encoder = SignalEncoder_1DTransformer_SE(
        #     input_channels=1,
        #     embedding_dim=self.modal_dim
        # )
        # self.signal_norm = nn.LayerNorm(self.modal_dim)

        # ✅ Clinical branch: TabNet encoder!
        self.clinical_encoder = ClinicalTabNetEncoder(
            input_dim=self.get_clinical_feature_dim(),
            latent_dim=32,
            device=config.device  # ✅ 멀티모달도 동일
        )
        self.load_pretrained_clinical_encoder('./checkpoints/clinical/best.pth')
        self.clinical_norm = nn.LayerNorm(self.clinical_dim)

        # ✅ Branch classifiers
        # self.image_classifier = nn.Linear(512, config.num_classes)
        # self.signal_classifier = nn.Linear(128, config.num_classes)
        # self.clinical_classifier = nn.Linear(32, config.num_classes)
        self.image_classifier = nn.Linear(self.image_dim, config.num_classes)
        self.signal_classifier = nn.Linear(self.signal_dim, config.num_classes)
        self.clinical_classifier = nn.Linear(self.clinical_dim, config.num_classes)

        # ✅ Attention Fusion + Fusion classifier
        # self.attention_fusion = AttentionFusion(dims=[512, 128, 32])
        # self.fusion_classifier = nn.Sequential(
        #     nn.Linear(512 + 128 + 32, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, config.num_classes)
        # )

        self.attention_fusion = AttentionFusion(dims=[self.image_dim, self.signal_dim, self.clinical_dim])
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.image_dim + self.signal_dim + self.clinical_dim, 128),
            # nn.Linear(self.modal_dim * 3, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes)
        )

    def get_clinical_feature_dim(self):
        return 2

    def load_pretrained_clinical_encoder(self, weight_path):
        self.clinical_encoder.load_pretrained_partial(weight_path)

    def load_pretrained_signal_encoder(self, weight_path, load_fc=False):
        checkpoint = torch.load(weight_path, map_location='cpu')
        if not load_fc:
            checkpoint = {
                k: v for k, v in checkpoint.items()
                # if not k.startswith('classifier')
                if not k.startswith('classifier.4')
            }
        missing, unexpected = self.signal_encoder.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded pretrained signal encoder from {weight_path}")
        if missing:
            print(f"⚠️  Missing keys: {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys: {unexpected}")

    def forward(self, image, ecg_signal, clinical):
        img_feat = self.image_encoder(image)  # [B, 512]
        img_feat = self.image_norm(img_feat)

        ecg_signal = ecg_signal.unsqueeze(1)
        signal_feat = self.signal_encoder(ecg_signal)
        signal_feat = self.signal_norm(signal_feat)

        clinical_feat,_ = self.clinical_encoder(clinical)
        clinical_feat = self.clinical_norm(clinical_feat)

        # 각 Modality의 Feature 분포 확인
        # print(f"Image feat: mean={img_feat.mean().item()}, std={img_feat.std().item()}")
        # print(f"Signal feat: mean={signal_feat.mean().item()}, std={signal_feat.std().item()}")
        # print(f"Clinical feat: mean={clinical_feat.mean().item()}, std={clinical_feat.std().item()}")

        img_logits = self.image_classifier(img_feat)
        signal_logits = self.signal_classifier(signal_feat)
        clinical_logits = self.clinical_classifier(clinical_feat)

        # fused = self.attention_fusion(img_feat, signal_feat, clinical_feat)
        fused, soft_weights = self.attention_fusion(img_feat, signal_feat, clinical_feat)
        fusion_logits = self.fusion_classifier(fused)

        # variance regularization (chunk-wise)
        var_img = torch.var(img_feat, dim=1).mean()
        var_signal = torch.var(signal_feat, dim=1).mean()
        var_clinical = torch.var(clinical_feat, dim=1).mean()
        var_loss = torch.abs(var_img - var_signal) + torch.abs(var_img - var_clinical) + torch.abs(
            var_signal - var_clinical)

        return img_logits, signal_logits, clinical_logits, fusion_logits, var_loss, soft_weights

    def load_pretrained_image_encoder(self, weight_path: str, load_fc: bool = False):
        """
        ✅ 이미지 전용으로 학습한 ResNet18 weight를 fusion model의 image branch에 로드
        :param weight_path: image-only classifier의 .pth 파일 경로
        :param load_fc: True면 fc 포함해서 가져옴, False면 fc 제외 (feature extractor만 가져옴)
        """
        print(f"🔄 Loading image encoder weights from {weight_path}")

        # 저장된 state_dict 로드
        saved_state = torch.load(weight_path, map_location='cpu')

        # 이미지 전용 모델: ImageOnlyClassifier 구조라고 가정
        tmp_resnet = models.resnet18()
        tmp_resnet.fc = nn.Linear(tmp_resnet.fc.in_features, self.image_dim)
        tmp_resnet.load_state_dict(saved_state, strict=False)

        # 가져올 state_dict 구성
        current_state = self.image_encoder.state_dict()
        new_state = tmp_resnet.state_dict()

        if not load_fc:
            # fc layer 제외 → feature extractor만 가져오기
            new_state = {k: v for k, v in new_state.items() if not k.startswith('fc.')}

        # 현재 모델에 업데이트
        current_state.update(new_state)
        self.image_encoder.load_state_dict(current_state)

        print(f"✅ Image encoder weights loaded! load_fc={load_fc}")