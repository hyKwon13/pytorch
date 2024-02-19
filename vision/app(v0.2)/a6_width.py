# 폭

import cv2
import numpy as np
from PySide2.QtGui import QImage, QPixmap
import time

def run_a6_width(cap, label):
# 마우스 이벤트 콜백 함수
    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, top_left_pt, bottom_right_pt

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            top_left_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bottom_right_pt = (x, y)
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

    # 웹캠 열기

    # 마우스 이벤트 초기화
    cv2.namedWindow('Vision Sensor')
    cv2.setMouseCallback('Vision Sensor', on_mouse)

    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    baseline_length = 0

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
        
        # 이미지 크기 변경 (선택사항)
        # frame = cv2.resize(frame, (640, 480))

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용 (노이즈 감소)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 케니 엣지 검출
        edges = cv2.Canny(blurred, 200, 50)

        # 선택한 영역에서만 컨투어 찾기
        if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
            cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

            roi = edges[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 직선 형태에 가까운 컨투어만 선택
            straight_contours = []
            for cnt in contours:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) <= 5:  # 다각형의 꼭짓점이 5개면 직선에 가까운 것으로 판단
                    straight_contours.append(cnt)

            sorted_contours = sorted(straight_contours, key=cv2.contourArea, reverse=True)

            # 가장 긴 두 개의 컨투어 선택
            longest_contours = sorted_contours[:2]

            # 컨투어 그리기 (선택사항)
            cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], longest_contours, -1, (0, 255, 0), 2)
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

            # 가운데 부분을 연결하는 직선 그리기
            if len(longest_contours) == 2:
                center1 = tuple(longest_contours[0].mean(axis=0, dtype=np.int)[0])
                center2 = tuple(longest_contours[1].mean(axis=0, dtype=np.int)[0])
                cv2.line(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], center1, center2, (0, 0, 255), 2)

                # 첫 프레임에서의 기준 길이 측정
                if baseline_length == 0:
                    baseline_length = np.linalg.norm(np.array(center1) - np.array(center2))

                # 컨투어의 길이 측정
                current_length = np.linalg.norm(np.array(center1) - np.array(center2))

                # 길이 비교 후 OK 또는 NG 표시
                if abs(current_length - baseline_length) < 10:  # 길이 차이가 10 픽셀 이내면 OK
                    cv2.putText(frame, 'OK', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8, cv2.LINE_AA)
                    cv2.line(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], center1, center2, (0, 255, 0), 2)
                    cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'NG', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8, cv2.LINE_AA)
                    cv2.line(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], center1, center2, (0, 0, 255), 2)
                    cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 0, 255), 2)
                    cv2.drawContours(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], longest_contours, -1, (0, 0, 255), 2)

            if not first_rectangle_drawn:
                # 첫 번째 사각형과 윤곽선을 frame0에 그립니다.
                cv2.rectangle(frame0, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
                cv2.drawContours(frame0[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], longest_contours, -1, (0, 255, 0), 2)
                cv2.line(frame0[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], center1, center2, (0, 255, 0), 2)
                first_rectangle_drawn = True  # 첫 번째 사각형이 그려졌다고 표시
            
            # 화면에 표시


        combined_frame = np.concatenate((frame0, frame), axis=1)

        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
                
        cv2.imshow('Vision Sensor', combined_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_a6_width()