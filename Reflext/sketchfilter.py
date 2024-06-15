import cv2
import numpy as np
import reflex as rx

# 이미지를 불러오고 그레이 스케일로 변환한다.
def sketchfilter(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 500))
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 블러 이미지를 만들고 블러 이미지와 그레이 스케일 이미지를 나눈다.
    BlurImg = cv2.GaussianBlur(GrayImg, (0, 0), 5)
    OutputImg = cv2.divide(GrayImg, BlurImg, scale=255)

    ret, buf = cv2.imencode('.jpg', OutputImg)
    byte_stream = buf.tobytes()

    return rx.image(src=rx.ByteStream(byte_stream))