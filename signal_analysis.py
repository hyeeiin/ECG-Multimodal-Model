import pandas as pd
import os

# signals 폴더 경로
signals_path = './data/signals/'

# error log
log_path = "./output/error_log.txt"
f = open(log_path, "a", encoding="utf-8")

# 모든 데이터를 담을 리스트
data_list = []

# 1.csv ~ 252.csv 반복
for i in range(1, 253):
    file_path = signals_path + f'{i}.csv'
    
    try:
        # CSV를 읽어서 1차원 배열로 변환
        df = pd.read_csv(file_path, header=None)
        flattened_row = df[0].values.flatten()  # column → row
        row_with_idx = [i] + flattened_row.tolist()
        data_list.append(row_with_idx)
    except:
        msg = f"number {i} is missing!"
        print(msg)
        f.write(f"{msg}\n")

# 모든 데이터를 DataFrame으로 합치기
merged_df = pd.DataFrame(data_list)
merged_df.columns = ['IDX'] + [f'col_{j}' for j in range(merged_df.shape[1] - 1)]

# 저장
merged_df.to_csv('./data/ecg_signals.csv', index=False, header=False)

f.close()

print(f'✅ signals 폴더의 244개 파일을 row로 합쳐서 저장: ecg_signals.csv')
