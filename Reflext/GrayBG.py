import cv2
import numpy as np

def apply_grabcut(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)

    # GrabCut 알고리즘에 필요한 임시 배열 생성
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 그랩컷
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 전경 배경 픽셀 설정
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

def create_foreground_background(image, mask):
    # 전경 이미지
    foreground = image * mask[:, :, np.newaxis]

    # 배경 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    background = background * (1 - mask)[:, :, np.newaxis]

    # 전경과 배경 합성
    result = cv2.add(foreground, background)
    return result

# 이미지 불러오기
image = cv2.imread('F:/Dev/Reflext/jjanggu3.jpg')

# 사각형 설정
rect = (1, 1, image.shape[1] , image.shape[0])

# GrabCut 알고리즘 적용
mask = apply_grabcut(image, rect)

# 배경을 흑백으로 변환하고 전경과 결합
result = create_foreground_background(image, mask)

# 결과 출력
cv2.imshow('Result Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()





# 'F:/Dev/Reflext/Reflext/test1.jpg'