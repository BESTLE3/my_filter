import cv2
import numpy as np

def apply_grabcut(image, rect):
    # 초기 마스크 생성
    mask = np.zeros(image.shape[:2], np.uint8)

    # GrabCut 알고리즘에 필요한 임시 배열 생성
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut 적용
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 확실한 배경과 전경 픽셀 설정
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

# 이미지 불러오기
image = cv2.imread('F:/Dev/Reflext/lesserafim2.jpg')
if image is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# 원본 이미지 크기 저장
height, width = image.shape[:2]

# 흑백 이미지 생성
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 그랩컷 알고리즘을 위한 초기 ROI 설정 (사각형으로 전경 영역 지정)
rect = (50, 50, width - 100, height - 100)  # 예제 사각형, 필요에 따라 조정

# GrabCut 알고리즘 적용하여 전경 추출
mask = apply_grabcut(image, rect)

# 흑백 이미지와 전경 분리 이미지를 합성
foreground = cv2.bitwise_and(image, image, mask=mask)
background = cv2.bitwise_and(gray_image, gray_image, mask=1 - mask)
result = cv2.add(foreground, background)

result = cv2.add(gray_image, mask)


# 결과 출력
# cv2.imshow('Foreground Extraction', foreground)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
