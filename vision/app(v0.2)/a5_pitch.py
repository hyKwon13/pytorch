# 피치검출

import cv2
import numpy as np
from PySide2.QtGui import QImage, QPixmap
import time

def run_a5_pitch(cap, label):

    # 마우스 이벤트 처리 함수
    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, top_left_pt, bottom_right_pt

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            top_left_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bottom_right_pt = (x, y)

    # 이미지 처리 및 출력 함수
    def process_and_display(frame, top_left_pt, bottom_right_pt):
        nonlocal first_rectangle_drawn
        # 선택한 영역을 초록색 사각형으로 표시
        roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 10, 70)

        # 초록색 선으로 윤곽선 그리기
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        
        contours_list.extend(contours)
        
        # 모든 윤곽선의 중심 연결
        for i in range(0, len(contours_list), 2):
            if i + 1 < len(contours_list):
                try:
                    average_contour = np.mean([contours_list[i], contours_list[i + 1]], axis=0).astype(int)
                    cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], [average_contour], -1, (0, 255, 0), 2)
                except ValueError as e:
                    return

        for i in range(len(contours_list) - 1):
            center1 = np.mean(contours_list[i], axis=0).astype(int)[0]
            center2 = np.mean(contours_list[i + 1], axis=0).astype(int)[0]
            cv2.line(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], tuple(center1), tuple(center2), (0, 255, 0), 2)
                        
        for i in range(1, len(contours_list)-2, 2):
            center1 = np.mean(contours_list[i], axis=0).astype(int)[0]
            center2 = np.mean(contours_list[i + 1], axis=0).astype(int)[0]
            # Calculate the Euclidean distance between two points
            length = np.linalg.norm(np.array(center2) - np.array(center1))

            # Store the line length in the list
            line_lengths.append(length)

        for i in range(0, len(line_lengths)):
            if abs(line_lengths[i] - line_lengths[0]) <= length_threshold:
                match_all = True
            else:
                match_all = False  # 일치하지 않는 경우 match_all을 False로 설정
                break

        if match_all:
            cv2.putText(frame, f"OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
        else:
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 0, 255), 2)
            cv2.putText(frame, f"NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)

            for i in range(0, len(contours_list), 2):
                if i + 1 < len(contours_list):
                    average_contour = np.mean([contours_list[i], contours_list[i + 1]], axis=0).astype(int)
                    cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], [average_contour], -1, (0, 0, 255), 2)
        
            
            for i in range(len(contours_list) - 1):
                center1 = np.mean(contours_list[i], axis=0).astype(int)[0]
                center2 = np.mean(contours_list[i + 1], axis=0).astype(int)[0]
                cv2.line(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], tuple(center1), tuple(center2), (0, 0, 255), 2)

        if not first_rectangle_drawn:
            # 첫 번째 사각형과 윤곽선을 frame0에 그립니다.
            cv2.rectangle(frame0, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            for i in range(0, len(contours_list), 2):
                if i + 1 < len(contours_list):
                    try:
                        average_contour = np.mean([contours_list[i], contours_list[i + 1]], axis=0).astype(int)
                        cv2.drawContours(frame0[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], [average_contour], -1, (0, 255, 0), 2)
                    except ValueError as e:
                        return

            for i in range(len(contours_list) - 1):
                center1 = np.mean(contours_list[i], axis=0).astype(int)[0]
                center2 = np.mean(contours_list[i + 1], axis=0).astype(int)[0]
                cv2.line(frame0[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], tuple(center1), tuple(center2), (0, 255, 0), 2)
            first_rectangle_drawn = True  # 첫 번째 사각형이 그려졌다고 표시
            
    # 카메라 초기화

    # 윈도우 생성 및 마우스 이벤트 콜백 함수 연결
    cv2.namedWindow('Vision Sensor')
    cv2.setMouseCallback('Vision Sensor', draw_rectangle)

    # 변수 초기화
    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    length_threshold = 5
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
        
        contours_list = []
        line_lengths = []

        
        # 선택한 영역이 있을 때만 이미지 처리 함수 호출
        if top_left_pt[0] != -1 and bottom_right_pt[0] != -1:
            process_and_display(frame, top_left_pt, bottom_right_pt)
            cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

        combined_frame = np.concatenate((frame0, frame), axis=1)

        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
                
        cv2.imshow('Vision Sensor', combined_frame)
        
        
        if cv2.waitKey(1) == 13:
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_a5_pitch()