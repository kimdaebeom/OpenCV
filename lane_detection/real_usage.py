import cv2
import numpy as np
import time

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)
def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
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
def roi(img): # Roi영역 설정
    vertices = np.array([[(0,height),(width, height), (width, height*1/3), (0,height*1/3)]], dtype=np.int32)
    roi_img = region_of_interest(img, vertices, (250,250,250))
    return roi_img
def draw_lines(img, lines, color=[250, 250, 250], thickness=1): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
def warping(img): # 버드아이뷰 적용
    src = np.float32([[170, 300], [470, 300], [0, 450], [640, 450]])
    dst = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_img = cv2.warpPerspective((img), matrix, (width, height))
    ret, minv = cv2.invert(matrix) #역행렬코드
    return bird_img, minv
    #M = cv2.getPerspectiveTransform(pts1,pts2)
    #warped = cv2.warpPerspective(img, M, (cols,rows))
    #ret, IM = cv2.invert(M)
    #restored = cv2.warpPerspective(warped, IM, (cols,rows))
def plothistogram(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    return leftbase, rightbase
def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    nwindows = 10
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장 
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값 
    margin = 50
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2
    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분

        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래


        win_xright_low = right_current - margin  # 왼쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 왼쪽 window 오른쪽 아래
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        #cv2.imshow("window", out_img)
        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))
    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)
    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 3) # y에 대한 x의 식으로3차함수로 근사(선형회귀??)
    else :
        left_fit = [0,0,0,0]
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 3)
    else :
        right_fit = [0,0,0,0]
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0]) # 0부터 높이-1 까지 높이만큼 간격으로 나눔(0에서 9까지 10간격으로 나누면 0 1 2 ... 9)
    left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3] # 3차함수로 식
    right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)
    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]
    cv2.imshow("window", out_img)
    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}
    return ret


def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    #restored = cv2.warpPerspective(bird_img, minv, (width, height))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)
    # 곡률 계산에 필요한 변수 및 픽셀값 조정
    img_height = warped_image.shape[0]
    y_eval = img_height
    ym_per_pix = 0.3 / 400.
    xm_per_pix = 0.845 / 640
    ploty = np.linspace(0, img_height - 1, img_height) # ploty를 새로운 polynomials로 초기화
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2) 
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    if left_fit_cr[0] == 0:
        left_curverad = 0
    else:
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0]) # 곡률 계산식
    if right_fit_cr[0] == 0:
        right_curverad = 0
    else:
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    if (left_fitx[0]>left_fitx[-1]) | (right_fitx[0]>right_fitx[-1]): # 열악한 영상 환경으로 인해 우회전, 조향을 결정할 선택차선을 고르기위한 조건문
        radius_sub = left_curverad
        if radius_sub == 0 :
            angle = 0
            curve_info = 'straight'
        else:
            angle = 180 / (2 * np.pi * radius_sub) # radian 형태의 곡률을 degree로 바꿔줌, 또한 수학적으로 조향각은 계산값에 2를 나눠주는 것이 타당
            curve_info = 'right_curve'
    elif (left_fitx[0]<left_fitx[-1]) | (right_fitx[0]<right_fitx[-1]): # 좌회전
        radius_sub = right_curverad
        if radius_sub == 0 :
            angle = 0
            curve_info = 'straight'
        else:
            angle = 180 / (2 * np.pi * radius_sub)
            curve_info = 'left_curve'
    else: # 직진
        angle = 0
        curve_info = 'straight'
    return result, angle, curve_info

cap = cv2.VideoCapture('track-s.mkv') # 동영상 불러오기

sec = 0.05 # time.sleep(sec)에 이용될 변수 선언

while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
         cap = cv2.VideoCapture('track-s.mkv')
         continue
    height, width = image.shape[:2] # 이미지 높이, 너비
    gray_img = grayscale(image) # 흑백이미지로 변환
    blur_img = gaussian_blur(gray_img, 3) # Blur 효과
    canny_img = canny(blur_img, 50, 220) # Canny edge 알고리즘
    roi_img = roi(canny_img)
    bird_img, minverse = warping(roi_img)
    #hough_img = hough_lines(bird_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
    leftv, rightv = plothistogram(bird_img) #히스토그램 생성
    draw_information = slide_window_search(bird_img,leftv,rightv) #슬라이딩윈도우 생성
    result, angle, curve_way = draw_lane_lines(image,bird_img,minverse,draw_information) #최종검출 차선 체크 및 조향각, 조향방향을 출력
    if angle > 49: # 조향각 49도로 제한
        angle = 49
    print ('way', curve_way, 'angle', np.round(angle))
    cv2.imshow('results',result)
    #time.sleep(sec) #반복문 속도 조절
    if cv2.waitKey(1) & 0xFF == ord('q'): #q입력시 창 종료
        break
# Release
cap.release()
cv2.destroyAllWindows()
