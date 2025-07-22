# ptb_xl_eda.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# === 1️⃣ Load CSV ===
df = pd.read_csv('./data/ptb_xl/ptbxl_database.csv')
df['scp_codes'] = df['scp_codes'].apply(eval)
df['label'] = df['scp_codes'].apply(
    lambda x: 0 if ('NORM' in x and float(x['NORM']) == 100.0) else 1
)

# === 2️⃣ Normal vs Abnormal 분포 ===
count_normal = (df['label'] == 0).sum()
count_abnormal = (df['label'] == 1).sum()

print(f"✅ Normal ECG Count: {count_normal}")
print(f"✅ Abnormal ECG Count: {count_abnormal}")
print(f"✅ Ratio Normal : Abnormal = {count_normal / (count_normal + count_abnormal):.2%} : {count_abnormal / (count_normal + count_abnormal):.2%}")

# === 3️⃣ AFIB 포함 비율 ===
# 'AFIB'가 scp_codes에 key로 존재하는 비율
df['AFIB'] = df['scp_codes'].apply(
    lambda x: 1 if ('AFIB' in x and float(x['AFIB']) == 100.0) else 0
)
count_afib = df['AFIB'].sum()
print(f"✅ AFIB Count: {count_afib}")
print(f"✅ AFIB Ratio: {count_afib / len(df):.2%}")

# === 4️⃣ Age & Weight 간단 통계 ===
print("\n=== Overall Age & Weight Stats ===")
print(df[['age', 'weight']].describe())

# === 5️⃣ Normal vs Abnormal 그룹별 통계 ===
group_normal = df[df['label'] == 1]
group_abnormal = df[df['label'] == 0]

# 나이
age_mean_normal = group_normal['age'].mean()
age_mean_abnormal = group_abnormal['age'].mean()
age_t, age_p = ttest_ind(group_normal['age'].dropna(), group_abnormal['age'].dropna(), equal_var=False)

# 체중
weight_mean_normal = group_normal['weight'].mean()
weight_mean_abnormal = group_abnormal['weight'].mean()
weight_t, weight_p = ttest_ind(group_normal['weight'].dropna(), group_abnormal['weight'].dropna(), equal_var=False)

print("\n=== Age Comparison ===")
print(f"Normal mean age: {age_mean_normal:.2f}")
print(f"Abnormal mean age: {age_mean_abnormal:.2f}")
print(f"t-test p-value: {age_p:.5f}")

print("\n=== Weight Comparison ===")
print(f"Normal mean weight: {weight_mean_normal:.2f}")
print(f"Abnormal mean weight: {weight_mean_abnormal:.2f}")
print(f"t-test p-value: {weight_p:.5f}")


# === 4 SARRH(부정맥) 포함 비율 ===
df['SARRH'] = df['scp_codes'].apply(
    lambda x: x if ('SARRH' in x and float(x['SARRH']) != 0.0) else 0
)
df_a = df['SARRH'][df['SARRH'] != 0].reset_index(drop=True)
df_a.dropna()

print(df_a)

# 5 기타 리듬 이상 -> abnormal
# 심장 리듬 이상으로 간주되는 코드 목록
df = pd.read_csv('./data/ptb_xl/ptbxl_database.csv')
abnormal_rhythm_codes = ['SR', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']

# 새로운 라벨링 로직
def label_ecg(scp_code_str):
    try:
        scp_dict = eval(scp_code_str)
        if ('AFIB' in scp_dict) and float(scp_dict['AFIB'] == 100.0):
            return 1  # AFIB
        elif any(code in scp_dict and float(scp_dict[code]) == 100.0 for code in abnormal_rhythm_codes):
            return 0  # Other abnormal rhythm
        else:
            return 2  # Other / ignore
    except:
        return 2  # malformed or unrecognized

# 라벨링 적용
df['label'] = df['scp_codes'].apply(label_ecg)

# 클래스별 개수 확인
label_counts = df['label'].value_counts().sort_index()
# label_map = {
#     0: 'Other abnormal rhythm (0)',
#     1: 'AFIB (1)',
#     2: 'Ignore (2)'
# }
# label_counts.index = [label_map.get(i, f'Unknown ({i})') for i in label_counts.index]
label_counts.index = ['Other abnormal rhythm (0)', 'AFIB (1)', 'Ignore (2)']
print(label_counts)
