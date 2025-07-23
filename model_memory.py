import torch
import os

from config import Config
from multimodal_paper_modal_balance import ECGMultimodalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[Before loading model] Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

model = ECGMultimodalModel(Config).to(device)

print(f"[After loading model] Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

model.load_state_dict(torch.load("./checkpoints/0716_173106/best.pth"))

print(f"[After loading model] Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")