from PIL import Image
import matplotlib.pyplot as plt

image_path = './data/images'

for i in range(1, 253):
    img = Image.open(f'{image_path}/{i}/{i:03d}ECG_lead2.jpg').convert('RGB')
    
    # 이미지 사이즈 출력
    print(f"Image {i}: size = {img.size}")  # (width, height) (2500,250)

    # 이미지 시각화
    plt.imshow(img)
    plt.title(f"Image {i}")
    plt.axis('off')  # 축 제거
    plt.show()
