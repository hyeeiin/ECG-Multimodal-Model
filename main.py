import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# ✅ 모델 정의
class CNN2D_LSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2D_LSTM, self).__init__()
        # CNN 블록
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 그레이스케일 1채널
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # LSTM 블록
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        # Fully Connected
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x)                   # (B, C, H, W)
        features = torch.mean(features, dim=2)   # (B, C, W) 평균풀링
        features = features.permute(0, 2, 1)     # (B, W, C)
        lstm_out, _ = self.lstm(features)        # (B, W, 128)
        out = lstm_out[:, -1, :]                 # 마지막 타임스텝
        out = self.fc(out)
        return out

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 512)),  # 긴 ECG 폭 유지
    transforms.Grayscale(),         # 흑백 변환
    transforms.ToTensor(),
])

# ✅ 테스트
img = Image.open("008ECG_lead2.jpg")
img = transform(img).unsqueeze(0)  # (1, 1, H, W)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2D_LSTM(num_classes=4).to(device)
output = model(img.to(device))
print("✅ Output logits:", output)
