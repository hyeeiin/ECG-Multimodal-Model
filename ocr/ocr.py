from google.cloud import vision
import io
import re
import os
import pandas as pd
import glob

# # 환경 설정
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./vision_key.json"
# client = vision.ImageAnnotatorClient()


# # OCR 함수
# def extract_text_from_image(image_path):
#     with io.open(image_path, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.Image(content=content)
#     response = client.text_detection(image=image)
#     texts = response.text_annotations
#     return texts[0].description if texts else ""

# # ✅ 모든 health_record{숫자}.png 파일 찾기
# image_files = sorted(glob.glob("health_image*.png"))

# # ✅ 전체 텍스트 결합
# full_text = ""
# for image_file in image_files:
#     print(f"🔍 OCR 처리 중: {image_file}")
#     text = extract_text_from_image(image_file)
#     full_text += "\n" + text

# print("--- OCR 결과 ---")
# # print(full_text)
# print("\n✅ 전체 OCR 결과 미리보기 (앞부분):\n")
# print(full_text[:1000])

# with open("ocr_result.txt", "w", encoding="utf-8") as f:
#     f.write(full_text)
# print("✅ OCR 결과가 ocr_result.txt에 저장되었습니다.")

with open("ocr_result.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# 변수 추출 함수
def extract_value_from_window(lines, index, window=3):
    for offset in range(-window, window + 1):
        i = index + offset
        if 0 <= i < len(lines):
            nums = re.findall(r"\d{1,3}\.?\d*", lines[i])
            if nums:
                return nums[0]
    return ""

def extract_values(text):
    result = {k: "" for k in [
        "연령", "성별", "수축기", "이완기",
        "흡연", "음주", "운동"
        "과거병력", "혈색소", "공복혈당", "총콜레스테롤", "고밀도 콜레스테롤", "중성지방", "저밀도 콜레스테롤",
        "AST", "ALT",  "감마지티피", "혈청 크레아티닌", "키", "몸무게"
    ]}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 연령 추출
    for line in lines:
        m = re.search(r"연령\s*[:：]?\s*(\d{1,3})\s*세", line)
        if m:
            result["연령"] = int(m.group(1))
            break

    # 성별 추출
    for line in lines:
        if "성별" in line:
            if "남" in line:
                result["성별"] = 0
            elif "여" in line:
                result["성별"] = 1
            break

    # 키/몸무게 추출
    for i, line in enumerate(lines):
        if "키" in line and ("몸무게" in line or "체중" in line):
            nums = []
            for j in range(1, 4):
                if i + j < len(lines):
                    nums += re.findall(r"\d{2,3}\.?\d*", lines[i + j])
            if len(nums) >= 2:
                result["키"] = nums[0]
                result["몸무게"] = nums[1]
                
    for i, line in enumerate(lines):
        if "고혈압" in line:
            for j in range(1, 4):
                if i + j < len(lines):
                    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", lines[i + j])
                    if m:
                        result["수축기"] = m.group(1)
                        result["이완기"] = m.group(2)
                        break
            break

    
    var_map = {
        "공복혈당": ["공복혈당", "혈당"],
        "총콜레스테롤": ["총콜레스테롤"],
        "고밀도 콜레스테롤": ["고밀도 콜레스테롤", "HDL"],
        "중성지방": ["중성지방"],
        "저밀도 콜레스테롤": ["저밀도 콜레스테롤", "LDL"],
        "AST": ["AST", "SGOT"],
        "ALT": ["ALT", "SGPT"],
        "혈색소": ["혈색소"],
        "혈청 크레아티닌": ["크레아티닌"],
        "감마지티피": ["감마지티피", "GTP"]
    }

    for var, keywords in var_map.items():
        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                val = extract_value_from_window(lines, i)
                if val:
                    result[var] = val
                    break

    # 혈색소 보정
    if result["혈색소"] and '.' not in result["혈색소"]:
        for line in lines:
            if "혈색소" in line:
                m = re.findall(r"\d{1,2}\.\d", line)
                if m:
                    result["혈색소"] = m[0]
                    break
    
    # ✅ 생활습관 항목 통합 추출
    lifestyle_keywords = {
        "흡연": "흡연",
        "음주": "음주",
        "운동": "운동"
    }

    current_section = None
    for line in lines:
        for key in lifestyle_keywords:
            if key in line:
                current_section = lifestyle_keywords[key]
                break

        if current_section and ("✅" in line or "■" in line or "☑" in line):
            result[current_section] = line.strip()
            current_section = None  # 다음으로 넘어감
    
    # 흡연
    if "과거 흡연자" in result["흡연"]:
        result["흡연"] = 1
    elif "현재 흡연자" in result["흡연"] or "전자담배" in result["흡연"]:
        result["흡연"] = 2
    elif "비흡연자" in result["흡연"]:
        result["흡연"] = 0

    # 음주
    if "비음주자" in result["음주"]:
        result["음주"] = 0
    elif any(word in result["음주"] for word in ["적정", "위험", "의심"]):
        result["음주"] = 1

    # 운동
    if result["운동"] is not None:
        if "건강증진" in result["운동"]:
            result["운동"] = 2
        elif any(word in result["운동"] for word in ["기본", "적절"]):
            result["운동"] = 1
        elif any(word in result["운동"] for word in ["부족"]):
            result["운동"] = 0

    return result



# 변수 추출
extracted = extract_values(full_text)

# 출력 및 저장
print("--- 추출된 변수 ---")
for k, v in extracted.items():
    print(f"{k}: {v}")

df = pd.DataFrame([extracted])
df.to_excel("GoogleOCR_추출결과.xlsx", index=False)
print("결과가 GoogleOCR_추출결과.xlsx에 저장되었습니다")
