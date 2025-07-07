import pandas as pd
import numpy as np

# === Load ===
df = pd.read_excel("./data/KDT2025_CRF_output.xlsx")  # 실제 파일 경로로!

# === 필터링 ===
exclude_idx = {17, 23, 36, 43, 51, 62, 115, 158}
df = df[~df['IDX'].isin(exclude_idx)]
df = df[df['ECG'] != 'Borderline'].reset_index(drop=True)

# === Drop MED_DT ===
df = df.drop(columns=['MED_DT'])

# === 연속형/이산형 분리 예시 ===
continuous_cols = ['AGE', 'Ht', 'Wt', 'SBP', 'DBP']
# 나머지 컬럼은 모두 binary/categorical로 처리한다고 가정
# binary_cols = [col for col in df.columns if col not in ['IDX', 'ECG'] + continuous_cols]
binary_cols = ['SMOKE','ALCHOL','PHY_ACT','HX_STROKE','HX_MI','HX_HTN','HX_DM','HX_DYSLI','HX_ATHERO','FHX_STROKE','FHX_MI','FHX_HTN','FHX_DM']

# === 함수 ===
def get_continuous_stats(series):
    mean = series.mean()
    std = series.std()
    return f"{mean:.1f}±{std:.1f}"

def get_binary_stats(series):
    pct = 100 * (series > 0).sum() / len(series)
    return f"{pct:.1f}"

def get_missing_pct(series):
    pct = 100 * series.isnull().sum() / len(series)
    return f"{pct:.1f}"

# === 세 그룹 ===
groups = {
    "All": df,
    "Abnormal": df[df['ECG'] == 'Abnormal'],
    "Normal": df[df['ECG'] == 'Normal']
}

# === 결과 ===
rows = []

for col in continuous_cols + binary_cols:
    row = {"Feature": col}
    for gname, gdf in groups.items():
        if col in continuous_cols:
            val = get_continuous_stats(gdf[col])
        else:
            print(f"{col}")
            val = get_binary_stats(gdf[col])
        miss = get_missing_pct(gdf[col])
        row[f"{gname}_Value"] = val
        row[f"{gname}_Miss"] = miss
    rows.append(row)

result = pd.DataFrame(rows)
print(result)

# === 저장 ===
# result.to_excel("clinical_stats_table.xlsx", index=False)

import matplotlib.pyplot as plt

# === result: 위에서 만든 clinical_stats_table DataFrame ===

fig, ax = plt.subplots(figsize=(12, len(result) * 0.5))

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Table
table = ax.table(
    cellText=result.values,
    colLabels=result.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # col, row scaling

# Save
plt.tight_layout()
# plt.savefig("clinical_stats_table.png", dpi=300)
plt.show()
# print("✅ Saved table as clinical_stats_table.png")

