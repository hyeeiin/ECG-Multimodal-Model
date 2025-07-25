import openai
import base64
import re
import pprint
import json

def main(image_path, abnormal, arrhythmia, af, age, sex, height, weight, smoke, alcohol, physical, hx, fhx):

    # === 1. API 키 설정 ===
    # openai.api_key = "key"  # 🔐 실제 키로 교체하세요

    # === 2. 이미지 → base64로 인코딩 ===
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    image_base64 = encode_image_to_base64(image_path)

    # clinical info
    abnormal = "정상" if abnormal == 0 else "비정상"
    arrhythmia = ", 특히 부정맥" if arrhythmia == 1 else ""
    af = ", 특히 심방세동" if af == 1 else ""
    sex = "여성" if sex == 1 else "남성"
    if smoke == 0:
        smoke = "비흡연자"
    elif smoke == 1:
        smoke = "과거 흡연자"
    else:
        smoke = "현재 흡연자"
    alcohol = "음주자" if alcohol == 1 else "비음주자"
    if physical == 0:
        physical = "운동 부족"
    elif physical == 1:
        physical = "저강도 운동"
    elif physical == 2:
        physical = "중강도 운동"
    else:
        physical = "고강도 운동"
    hx_text = ""
    if len(hx) != 0:
        for i in range(len(hx)):
            hx_text += hx[i]
            if i != len(hx) -1:
                hx_text += ", "
        hx_text += "의 과거력이 있음."
    fhx_text = ""
    if len(fhx) != 0:
        for i in range(len(fhx)):
            fhx_text += fhx[i]
            if i != len(fhx) -1:
                fhx_text += ", "
        fhx_text += "의 가족력이 있음."

    # === 3. 시스템 지시 및 유저 메시지 구성 ===
    system_prompt = "너는 심전도를 해석할 수 있는 임상 의사이며 ECG 전문가야."

    user_prompt = f"""
    해당 ECG 이미지를 보고 모델이 {abnormal}{arrhythmia}{af}이라고 판단한 것을 Grad-CAM을 통해서 어디 부분을 보았는지 heatmap으로 표현한 거야.
    이 heatmap을 근거로, ECG 파형 중 어떤 부분(RR 간격, QRS 파형, T파, P파 등)에 주목했는지 설명하고, 해석 결과를 기반으로 임상적으로 의미 있는 판단을 내려줘.

    다음 환자 정보도 함께 고려해서 해석해줘:
    - 나이: {age}세
    - 성별: {sex}
    - 키: {height}cm
    - 몸무게: {weight}kg
    - 흡연 여부: {smoke}
    - 음주 여부: {alcohol}
    - 신체 활동: {physical}
    - 병력: {hx_text}
    - 가족력: {fhx_text}

    아래와 같은 **형식만 참고**해서 작성해줘. 실제 내용은 Grad-CAM 이미지와 환자 정보를 기반으로 새롭게 생성해줘:

    예시 형식 (형식만 참고, 내용 복붙 금지):

    ## 🧠 Grad-CAM + 환자 정보 해석

    ### [RR 간격]

    (Grad-CAM에서 RR 간격과 관련된 이상 여부 + 임상적 해석)

    ---

    ### [QRS 파형]

    (QRS의 이상 여부 및 그 임상적 의미)

    ---

    ### [T파]

    (T파에 대한 해석 및 전해질 이상, 재분극 장애 가능성 등)

    ---

    ### [P파]

    (P파의 명확성 여부 및 동성 리듬 여부 판단)

    ---

    ### [임상 권고]

    - (Holter 등 추가 검사)
    - (심초음파 또는 전해질 패널 검사)
    - (특정 치료나 운동 조절 권고 등)
    """

    print(user_prompt)

    # === 4. GPT-4-vision API 호출 (GPT-4 Turbo with vision) ===
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=2048,
    )

    # === 5. 출력 ===
    print(response["choices"][0]["message"]["content"])
    gpt_output = response["choices"][0]["message"]["content"]

    # 추출할 섹션 제목
    sections = ["RR 간격", "QRS 파형", "T파", "P파", "임상 권고"]

    # 정규표현식 패턴 생성
    pattern = r"### \[(" + "|".join(sections) + r")\]\n(.*?)(?=\n### \[|\Z)"
    matches = re.findall(pattern, gpt_output, re.DOTALL)

    # 딕셔너리 구성
    section_dict = {section: "" for section in sections}
    for name, content in matches:
        section_dict[name] = content.strip().strip("---").strip()

    # 결과 출력 및 JSON 저장
    pprint.pprint(section_dict)

    # JSON 저장 (선택)
    with open("ecg_analysis_15.json", "w", encoding="utf-8") as f:
        json.dump(section_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    image_path = "./abnormal_gradcam_15_overlay.png"
    abnormal = 1
    arrhythmia = 1
    af = 0
    age = 84
    sex = 1
    height = 143.8
    weight = 43.3
    smoke = 0
    alcohol = 0
    physical = 2
    hx = ["뇌졸중", "고혈압"]
    fhx = []

    main(image_path, abnormal, arrhythmia, af, age, sex, height, weight, smoke, alcohol, physical, hx, fhx)