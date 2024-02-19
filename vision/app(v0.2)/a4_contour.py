# 윤곽

import cv2
import numpy as np
from PySide2.QtGui import QImage, QPixmap
import time

def run_a4_contour(cap, label):
    # 마우스 이벤트 처리 함수
    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, top_left_pt, bottom_right_pt, reference_contours

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            top_left_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bottom_right_pt = (x, y)
            # 선택한 영역의 윤곽선 추출
            roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            reference_contours = extract_contours(roi)
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

    # 윤곽선 추출 함수
    def extract_contours(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)  # bilateral filter 적용
        canny = cv2.Canny(blur, 50, 150)  # 동적 임계값 사용
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # 이미지 처리 및 출력 함수
    def process_and_display(frame, top_left_pt, bottom_right_pt):
        nonlocal reference_contours, first_rectangle_drawn
        
        if not first_rectangle_drawn:
            # 첫 번째 사각형과 윤곽선을 frame0에 그립니다.
            cv2.rectangle(frame0, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            cv2.drawContours(frame0[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], reference_contours, -1, (0, 255, 0), 2)
            first_rectangle_drawn = True  # 첫 번째 사각형이 그려졌다고 표시
            
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
        contours = extract_contours(roi)

        result = compare_contours(reference_contours, contours)
        
        if not result:
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 0, 255), 2)
            cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], reference_contours, -1, (0, 0, 255), 2)
            cv2.putText(frame, "NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
        else:
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], reference_contours, -1, (0, 255, 0), 2)
            cv2.putText(frame, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

    # 윤곽선 비교 함수
    def compare_contours(contours1, contours2):
        if not contours1 or not contours2:
            return False
        # 윤곽선 간 최소 거리 비교
        best_match_score = float('inf')
        for c1 in contours1:
            for c2 in contours2:
                match_score = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
                best_match_score = min(best_match_score, match_score)
        threshold = 0.1
        return best_match_score < threshold

    cv2.namedWindow('Vision Sensor')
    cv2.setMouseCallback('Vision Sensor', draw_rectangle)

    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    reference_contours = []
    first_rectangle_drawn = False
        
    # 초당 프레임 수를 측정하기 위한 변수
    start_time = time.time()
    frame_count = 0
    fps = 0  # 초기화된 fps 변수 추가

   # 처음 몇 프레임은 무시하기 위해 루프 수행
    for _ in range(30):  # 예시로 처음 30프레임을 무시하도록 설정 (적절한 수를 선택)
        _, _ = cap.read()

    ret, frame0 = cap.read()
    
    while True:
        ret, frame = cap.read()

        # 초당 프레임 수를 증가시킴
        frame_count += 1

        # 현재 시간을 갱신하여 1초가 경과했는지 확인
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 1.0:
            # 1초가 경과하면 초당 프레임 수를 출력하고 변수 초기화
            fps = frame_count / elapsed_time

            start_time = current_time
            frame_count = 0
            
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not ret:
            break

        if top_left_pt[0] != -1 and bottom_right_pt[0] != -1 and not drawing:
            process_and_display(frame, top_left_pt, bottom_right_pt)
            cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

        combined_frame = np.concatenate((frame0, frame), axis=1)

        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
                
        cv2.imshow('Vision Sensor', combined_frame)

        if cv2.waitKey(1) == 13:  # Enter key to break
            break

    cap.release()
    cv2.destroyAllWindows()