import cv2

# 이미지 불러오고 크기 조절 후 높이와 너비를 구해준 후 이미지는 작게 만듬
# 좀 더 과장되게 표현하기 위해 높이와 너비를 3로 나누어줌.
img = cv2.imread('hoochooimg/IMG_5723.jpg')  
img = cv2.resize(img, (400, 500))
h, w = img.shape[:2]
img2 = cv2.resize(img, (w//2, h//2))

# 크기 조절한 이미지(img2)를 양방향 필터링으로 에지가 아닌 부분만 블러링
# 크기 조절한 이미지(img2)를 캐니 에지 검출기로 에지 검출 후 255에서 빼주어 흰 부분과 검은 부분 반전
# 에지 검출한 이미지를 그레이 스케일 이미지로 변환
BlurImg = cv2.bilateralFilter(img2, -1, 10, 1)
EdgeImg = 255 - cv2.Canny(img2, 100, 100)
EdgeImg = cv2.cvtColor(EdgeImg, cv2.COLOR_GRAY2BGR)

# 블러링한 이미지와 에지 검출한 이미지를 AND연산으로 결합
# 처음에 크기를 조절했던 이미지를 원래 크기로 조정
OutputImg = cv2.bitwise_and(BlurImg, EdgeImg)
OutputImg = cv2.resize(OutputImg, (w,h), interpolation=cv2.INTER_NEAREST)

cv2.imshow('PixelImg', OutputImg)

cv2.waitKey(0)
cv2.destroyAllWindows()