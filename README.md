# 🫀 ECG Multimodal Classification

This project implements a **multimodal deep learning model** to classify ECG signals as **normal** or **abnormal** using:
- **ECG images**
- **ECG time-series signal (digitized)**
- **Clinical metadata**

---

## 📂 Project Structure

ECG-Multimodal-Model/
├── data/
      ├── images/
            ├── 1/
                ├── 001ECG_lead2.jpg
            ├── 2/
               ...
      ├── ecg_signals.csv
      ├── clinical.xlsx
      └── labels.xlsx
├── config.py
├── dataset.py
├── multimodal.py
├── train.py
├── checkpoints/
├── runs/
└── output/


---

## 📄 Data

- **images/**: Folder per patient, containing ECG images (`2500×250`).
- **labels.xlsx**: Contains `index` and `label` (`normal`, `abnormal`, `borderline`).
  - `borderline` samples are **excluded** during training.
- **ecg_signals.csv**: Digitized ECG time-series (rows: index, columns: time points).
- **clinical.xlsx**: Includes `IDX` column for index and patient clinical metadata (e.g., age, sex, blood pressure, diabetes history).

All modalities must have **matching patient indices**. Missing indices (e.g., 17, 23, 36, ...) are automatically excluded.

---

## ⚙️ Key Files

| File | Description |
|------|-------------|
| `config.py` | Configuration for data paths, image size, hyperparameters, and device. |
| `dataset.py` | Loads all modalities, filters for common indices, excludes `borderline`, applies Z-score normalization, and creates stratified splits (train:val:test = 8:1:1). |
| `multimodal.py` | The multimodal model: ResNet18 for images (pretrained), MLP for ECG signals, MLP for clinical data, with feature fusion. |
| `train.py` | Full training pipeline with early stopping, dynamic learning rate adjustment, and TensorBoard logging. |

---

## 🚀 Usage

### 1️⃣ Install environment

```bash
conda create -n ecgmm python=3.10
conda activate ecgmm
pip install pandas scikit-learn tqdm tensorboard
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2️⃣ Run TensorBoard
```bash
tensorboard --logdir=runs
```
http://localhost:6006/

### 3️⃣ Train
```bash
python train.py
```

## 📝 Reference

This project is inspired by:

**Anatomy-Informed Multimodal Learning for Myocardial Infarction Prediction**  
Ivan-Daniel Sievering, Ortal Senouf, et al.  
*IEEE Open Journal of Engineering in Medicine and Biology, 2024.*  
[Read on PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC11573417/)
