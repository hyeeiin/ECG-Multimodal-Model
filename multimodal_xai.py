import torch
import torch.nn.functional as F
import FinalProject.shap_fusion as shap_fusion
import matplotlib.pyplot as plt
from torch.autograd import Variable
from config import Config
from dataset import get_testloader
from multimodal import ECGMultimodalModel


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, ecg_signal, clinical, class_idx=None):
        # forward pass
        output = self.model(input_image, ecg_signal, clinical)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        # one_hot[0][class_idx] = 1
        for i in range(output.shape[0]):
            one_hot[i][class_idx[i]] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 계산
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))  # GAP

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()

        return cam.cpu().numpy()


def explain_ecg_signal(model, background, ecg_signal_sample):
    explainer = shap_fusion.DeepExplainer(model.signal_encoder, background)
    shap_values = explainer.shap_values(ecg_signal_sample)
    shap_fusion.summary_plot(shap_values, ecg_signal_sample.cpu().numpy())


def explain_clinical(model, background, clinical_sample):
    explainer = shap_fusion.Explainer(model.clinical_encoder, background)
    shap_values = explainer(clinical_sample)
    shap_fusion.plots.waterfall(shap_values[0])


def run_xai_example(model, image, ecg_signal, clinical, device):
    model.eval()

    image = image.to(device)
    ecg_signal = ecg_signal.to(device)
    clinical = clinical.to(device)

    # === Grad-CAM ===
    target_layer = model.image_encoder.layer4[-1]  # ResNet 마지막 블록
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(image, ecg_signal, clinical)

    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM (Image branch)")
    plt.show()

    # === SHAP ECG ===
    print("\n✅ ECG signal SHAP")
    background_ecg = ecg_signal[:10]  # 샘플링
    explain_ecg_signal(model, background_ecg, ecg_signal[:1])

    # === SHAP Clinical ===
    print("\n✅ Clinical SHAP")
    background_clinical = clinical[:10]
    explain_clinical(model, background_clinical, clinical[:1])


if __name__ == "__main__":

    device = torch.device(Config.device)

    # Load model
    model = ECGMultimodalModel(Config).to(device)
    ckpt_path = "./checkpoints/0708_094809/best_epoch2.pth"  # 🔑 학습된 가중치로 변경!
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # ✅ 너가 지정한 index
    test_indices = [101, 173, 136, 109, 184, 2, 201, 11, 163, 149, 218, 50, 135, 83, 142, 140, 206, 6, 108, 197, 154, 112]
    test_loader = get_testloader(Config, test_indices)

    # _, _, test_loader = get_dataloaders(Config)
    batch = next(iter(test_loader))
    image, ecg_signal, clinical, label = batch

    run_xai_example(model, image, ecg_signal, clinical, device)
