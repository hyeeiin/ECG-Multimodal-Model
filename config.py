# config.py

import os
import torch

class Config:
    # Random seed
    seed = 42

    # Image input
    img_height = 224
    img_width = 224

    # Data paths
    data_dir = "./data"
    image_dir = os.path.join(data_dir, "images")
    ecg_csv = os.path.join(data_dir, "ecg_signals.csv")
    label_file = os.path.join(data_dir, "labels.xlsx")
    clinical_file = os.path.join(data_dir, "clinical.xlsx")

    # Model
    num_classes = 2  # normal vs abnormal

    # Training hyperparameters
    batch_size = 8
    num_epochs = 50
    lr = 1e-4
    patience = 5

    # Checkpoint dir
    checkpoint_dir = "./checkpoints"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
