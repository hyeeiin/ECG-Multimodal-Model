# config.py
import torch

class Config:
    # Paths
    IMAGE_DIR = "./data/images/"
    CLINICAL_CSV = "./data/clinical_data.csv"
    ECG_CSV = "./data/ecg_data.csv"
    SIGNAL_DIR = "./data/signals/"
    CHECKPOINT_DIR = "./checkpoints/"
    
    # Data
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 3  # Normal / Abnormal / Borderline
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    VALID_SPLIT = 0.2
    
    # Model
    PRETRAINED = True
    DROPOUT = 0.3
    
    # Training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    CLINICAL_INPUT_DIM = 15  # 나이, 성별 등 feature 수
    ECG_INPUT_DIM = 5000     # ECG 시계열 길이
    SEED = 42
    PATIENCE = 5  # Early stopping patience

    # Loss weights
    LAMBDA_IMAGE = 0.3
    LAMBDA_CLINICAL = 0.3
    LAMBDA_ECG = 0.2
    LAMBDA_FUSION = 0.2

config = Config()
