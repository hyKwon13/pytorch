# 직경

import cv2
import numpy as np
from PySide2.QtGui import QImage, QPixmap
import time

def run_a7_diameter(cap, label):

    selected_roi = None
    roi_selected = False
    top_left = None
    bottom_right = None
    reference_diameter = None  # 첫 프레임에서 얻은 diameter를 저장할 변수

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_roi, roi_selected, top_left, bottom_right

        frame = param

        if event == cv2.EVENT_LBUTTONDOWN:
            selected_roi = [(x, y)]
            roi_selected = False

        elif event == cv2.EVENT_LBUTTONUP:
            selected_roi.append((x, y))
            roi_selected = True
            top_left = selected_roi[0]
            bottom_right = selected_roi[1]

            # 영역을 직사각형으로 그리기
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    def measure_diameter(frame, center, radius):
        # 이미지에서 원의 직경을 측정
        diameter = radius * 2

        # 지름을 가르는 선 그리기
        cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 0), 2)

        return diameter


        # 웹캠 열기

    cv2.namedWindow('Vision Sensor')
    cv2.setMouseCallback('Vision Sensor', on_mouse, param=cap)

    # 초당 프레임 수를 측정하기 위한 변수
    start_time = time.time()
    frame_count = 0
    fps = 0  # 초기화된 fps 변수 추가
    first_rectangle_drawn = False
    # 처음 몇 프레임은 무시하기 위해 루프 수행
    for _ in range(30):  # 예시로 처음 30프레임을 무시하도록 설정 (적절한 수를 선택)
        _, _ = cap.read()

    ret, frame0 = cap.read()
    
    while True:
        # 프레임 읽기
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
            
    
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        if roi_selected:
            cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
            # 선택된 영역을 그레이스케일로 변환
            gray_roi = cv2.cvtColor(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], cv2.COLOR_BGR2GRAY)

            # 영역을 블러 처리하여 노이즈 제거
            blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            # 캐니 엣지 검출
            edges_roi = cv2.Canny(blurred_roi, 50, 150)

            # 원 검출
            circles = cv2.HoughCircles(edges_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=50)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0] + top_left[0], i[1] + top_left[1])
                    radius = i[2]

                    # 초록색 원 그리기 (전체 프레임에 그리는 것이 아니라 roi에 그립니다.)
                    cv2.circle(frame, (center[0], center[1]), radius, (0, 255, 0), 2)

                    # 직경 측정
                    diameter = measure_diameter(frame, center, radius)

                    # 첫 프레임에서 얻은 diameter를 기준으로 +-5 범위 설정
                    if reference_diameter is None:
                        reference_diameter = diameter

                    if not first_rectangle_drawn:
                        # 첫 번째 사각형과 윤곽선을 frame0에 그립니다.
                        cv2.rectangle(frame0, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.circle(frame0, (center[0], center[1]), radius, (0, 255, 0), 2)
                        cv2.line(frame0, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 0), 2)
                        first_rectangle_drawn = True  # 첫 번째 사각형이 그려졌다고 표시

                    # +- 5 범위 확인 및 결과 표시
                    if reference_diameter - 5 <= diameter <= reference_diameter + 5:
                        result_text = "OK"
                        cv2.putText(frame, f"{result_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.circle(frame, (center[0], center[1]), radius, (0, 255, 0), 2)

                    else:
                        result_text = "NG"
                        cv2.putText(frame, f"{result_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
                        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                        cv2.circle(frame, (center[0], center[1]), radius, (0, 0, 255), 2)
                        cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 255), 2)
                

        combined_frame = np.concatenate((frame0, frame), axis=1)

        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
                
        cv2.imshow('Vision Sensor', combined_frame)

        # 종료 키 확인
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    run_a7_diameter()