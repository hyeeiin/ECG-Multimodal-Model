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
    clinical_file = os.path.join(data_dir, "clinical.csv")

    af_label_file = os.path.join(data_dir, "af_labels.xlsx")
    
    # Physionet data paths
    physionet_dir = "./data/physionet"
    physionet_data_dir = "./data/physionet/training2017"
    physionet_label_file = os.path.join(physionet_dir, "REFERENCE.csv")

    # Model
    num_classes = 2  # normal vs abnormal

    # Training hyperparameters
    batch_size = 16
    num_epochs = 30
    lr = 1e-4 # image: 1e-4, signal: 1e-3
    patience = 5

    # CV Settings
    k_outer = 5
    k_inner = 3

    # Checkpoint dir
    checkpoint_dir = "./checkpoints"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
