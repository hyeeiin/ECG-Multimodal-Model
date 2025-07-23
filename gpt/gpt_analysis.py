import openai
import base64
from pathlib import Path

def main():

    # === 1. API 키 설정 ===
    # openai.api_key = "KEY"  # 🔐 실제 키로 교체하세요

    # === 2. 이미지 → base64로 인코딩 ===
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    image_path = "./abnormal_gradcam_65.png"
    image_base64 = encode_image_to_base64(image_path)

    # === 3. 시스템 지시 및 유저 메시지 구성 ===
    system_prompt = "너는 심전도를 해석할 수 있는 임상 의사이며 ECG 전문가야."

    user_prompt = """
    해당 ECG 이미지를 보고 모델이 비정상이라고 판단한 것을 Grad-Cam을 통해서 어디 부분을 보았는지 heatmap으로 표현한 거야.
    이걸 가지고 ECG 신호의 어떤 부분을 보았는지 해석할 수 있을까? (예. 불규칙한 간격, 정상인 QRS 파형 등)

    환자 정보는 다음과 같아:
    83세의 여성, 키 142.1cm, 몸무게 45.5kg, 흡연·음주는 하지 않음, 고강도 운동을 즐기고 뇌졸중의 과거력이 있음.

    예시 형식:
    ## 🧠 Grad-CAM + 환자 정보 해석

    ### 1. 🔁 **RR 간격이 일정하지 않음**
    ...

    ### 2. 🧡 **QRS 폭 강조**
    ...

    ## 🧭 임상 권고
    ...
    """

    # === 4. GPT-4-vision API 호출 (GPT-4 Turbo with vision) ===
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
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
        max_tokens=1024,
    )

    # === 5. 출력 ===
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    abnormal = 1
    age = 83
    sex = 1
    height = 142.1
    weight = 45.5
    smoke = 0
    alcohol = 0

    main()