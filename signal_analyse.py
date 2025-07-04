import pandas as pd
import os

# signals 폴더 경로
signals_path = './data/signals/'

# 첫 번째 파일로 데이터 길이 파악
first_df = pd.read_csv(signals_path +'1.csv', header=None)
num_rows = len(first_df)

# 빈 DataFrame 생성 (row는 CSV의 row 수, column은 파일 개수)
merged_df = pd.DataFrame(index=range(num_rows))

# error log
log_path = "./output/error_log.txt"
f = open(log_path, "a", encoding="utf-8")

# 각 CSV를 열로 추가
for i in range(1, 253):
    try:
        file_path = f"{signals_path}{i}.csv"
        df = pd.read_csv(file_path, header=None)
        
        # 컬럼 이름을 파일 번호로 지정
        merged_df[i] = df[0]
    except:
        msg = f"number {i} is missing!"
        print(msg)
        f.write(msg + "\n")

# 결과 저장
merged_df.to_csv('./data/ecg_signals.csv', index=False)

f.close()

print(f'✅ signals 폴더의 244개 파일을 하나의 CSV로 병합 완료: ecg_signals.csv')
