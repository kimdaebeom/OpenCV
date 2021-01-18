import cv2 # opencv 사용
import numpy as np

#허프까지만

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def adaptive(img):
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5)
def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)
def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅
    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image
def draw_lines(img, lines, color=[250, 250, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
cap = cv2.VideoCapture('track-s.mkv') # 동영상 불러오기
while(cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2] # 이미지 높이, 너비
    gray_img = grayscale(image) # 흑백이미지로 변환
    #test_img = adaptive(gray_img)
    blur_img = gaussian_blur(gray_img, 3) # Blur 효과
    canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘
    # 사다리꼴 모형의 Points
    vertices = np.array([[(0,height),(width, height), (width, height*2/3), (0,height*2/3)]], dtype=np.int32)
    roi_img = region_of_interest(canny_img, vertices, (250,250,250)) # vertices에 정한 점들 기준으로 ROI 이미지 생성
    #hough_img = hough_lines(roi_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
    cv2.imshow('results', roi_img) # 이미지 출력
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release
cap.release()
cv2.destroyAllWindows()

