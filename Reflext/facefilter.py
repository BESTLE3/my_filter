import cv2
import numpy as np

face_cascade_path = 'F:/Dev/Reflext/haarcascade_frontalface_default.xml'
eye_cascade_path = 'F:/Dev/Reflext/haarcascade_eye.xml'
image_path = 'Reflext/test1.jpg'
overlay_image_path = 'c:/Users/whwhd/Downloads/pngegg.png'

image = cv2.imread(image_path)
overlay_image = cv2.imread(overlay_image_path)

# Haar Cascade는 그레이 스케일 이미지에서 얼굴을 검출함
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 얼굴과 눈 검출용 CascadeClassifier 객체
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
# detectMultiScale 함수로 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
# 검출된 얼굴과 눈 주변에 사각형 그리기 (색상[0, 255, 0], 두께 : 3)

# Iterate over the detected faces
for (x, y, w, h) in faces:
    resized_overlay_image = cv2.resize(overlay_image, (w, h))
    
    # 원본 이미지의 해당 영역에 삽입
    image[y:y+h, x:x+w] = resized_overlay_image
    

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # 눈 주변에 사각형 그리기
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
# 결과 이미지 출력
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()