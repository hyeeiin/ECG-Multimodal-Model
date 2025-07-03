# split dataset
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def stratifySplit(split_ratios, csv_path, data_dir, output_dir):
    """
    Stratified split of data into train, val, test folders.

    Args:
        split_ratios (list): [train_ratio, val_ratio, test_ratio], e.g., [0.7, 0.2, 0.1]
        csv_path (str): Path to CSV file containing index and labels.
        data_dir (str): Root directory containing data/<index>/*.jpeg
        output_dir (str): Root directory where train/val/test folders will be created.
    """
    assert sum(split_ratios) == 1.0, "split_ratios must sum to 1.0"
    train_ratio, val_ratio, test_ratio = split_ratios

    # 1. Load CSV
    df = pd.read_csv(csv_path)  # assumes columns: index,label
    print(f"Total samples: {len(df)}")

    # 2. Train/Val/Test Split
    train_df, temp_df = train_test_split(
        df, test_size=(1-train_ratio), stratify=df['label'], random_state=42
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)  # Adjust ratio for temp split
    val_df, test_df = train_test_split(
        temp_df, test_size=(1-relative_val_ratio), stratify=temp_df['label'], random_state=42
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Copy files to output_dir/train, output_dir/val, output_dir/test
    for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        split_path = os.path.join(output_dir, split_name)
        os.makedirs(split_path, exist_ok=True)

        for idx in split_df['index']:
            src_dir = os.path.join(data_dir, str(idx))
            dst_dir = os.path.join(split_path, str(idx))
            shutil.copytree(src_dir, dst_dir)

        print(f"Copied {len(split_df)} samples to {split_path}")

    print("✅ Stratified split completed.")

    return None

# # 사용 예시
# split_ratios = [0.7, 0.2, 0.1]
# csv_path = "labels.csv"           # index,label 형태
# data_dir = "data"                 # data/1/*.jpeg 형태
# output_dir = "output"             # 결과 train/val/test 저장

# stratifySplit(split_ratios, csv_path, data_dir, output_dir)



# load dataset
# 1. Image data
def get2DImageData():




    return None