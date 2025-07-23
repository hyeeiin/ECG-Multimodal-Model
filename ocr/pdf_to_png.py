from google.cloud import vision
from pdf2image import convert_from_path
import io
import re
import os
import pandas as pd

pdf_path = "./health_record.pdf"
image_output_path = "./health_image"
images = convert_from_path(pdf_path, dpi=300)
for i in range(len(images)):
    images[i].save(image_output_path + f"{i}.png", "PNG")
    print(f"✅ PDF에서 이미지 추출 완료: {image_output_path}")

image_path = "./health_image0.png"
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)