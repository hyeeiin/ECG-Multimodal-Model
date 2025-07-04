# config.py

import os
import torch

class Config:
    seed = 42

    # 이미지
    img_height = 224
    img_width = 224
    img_channels = 3

    # 시계열 ECG 수치 데이터
    num_ecg_features = 244

    # 임상 데이터: (필드 확인 후 숫자 맞춰주세요)
    num_clinical_features = 18

    # 클래스 수
    num_classes = 2  # normal vs abnormal

    # 데이터 경로
    data_dir = "./data"
    image_dir = os.path.join(data_dir, "images")
    ecg_csv = os.path.join(data_dir, "ecg_signals.csv")
    label_file = os.path.join(data_dir, "labels.xlsx")
    clinical_file = os.path.join(data_dir, "clinical.xlsx")

    # 학습
    batch_size = 16
    num_epochs = 50
    lr = 1e-4
    patience = 5

    # loss weights (멀티로 할 경우)
    LAMBDA_IMAGE = 0.25
    LAMBDA_ECG = 0.25
    LAMBDA_CLINICAL = 0.25
    LAMBDA_FUSION = 0.25

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoints
    checkpoint_dir = "./checkpoints"
