# 돌출 검출

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import time
from PySide2.QtGui import QImage, QPixmap

def run_ssim(cap, label):
    # 마우스 이벤트 콜백 함수
    def draw_rectangle(event, x, y, flags, param):
        nonlocal rect_count, rectangles_left, rectangles_right, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            rectangles_left[rect_count] = [(x, y)]
            rectangles_right[rect_count] = [(x, y)]  # 오른쪽 창의 리스트도 초기화
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            rectangles_left[rect_count].append((x, y))
            rectangles_right[rect_count].append((x, y))
            drawing = False
            rect_count += 1

        elif event == cv2.EVENT_RBUTTONDOWN and rect_count > 0:
            # 마우스 우클릭으로 마지막으로 추가한 영역 삭제
            rect_count -= 1
            rectangles_left.pop(rect_count, None)
            rectangles_right.pop(rect_count, None)

    rect_count = 0  # 영역 개수
    rectangles_left = {}  # 왼쪽 창의 각 영역의 좌표를 저장하는 딕셔너리
    rectangles_right = {}  # 오른쪽 창의 각 영역의 좌표를 저장하는 딕셔너리
    drawing = False  # 마우스 드래그 중 여부

    # 처음 몇 프레임은 무시하기 위해 루프 수행
    for _ in range(30):  # 예시로 처음 30프레임을 무시하도록 설정 (적절한 수를 선택)
        _, _ = cap.read()

    ret, frame1 = cap.read()

    # 창 생성
    cv2.namedWindow('Vision Sensor')

    # 마우스 이벤트 콜백 등록
    cv2.setMouseCallback('Vision Sensor', draw_rectangle)

    # 초당 프레임 수를 측정하기 위한 변수
    start_time = time.time()
    frame_count = 0
    fps = 0  # 초기화된 fps 변수 추가

    while True:
        # 첫 번째 영역의 프레임을 받아옴
        ret, frame2 = cap.read()

        # 왼쪽 창과 오른쪽 창에 현재까지 그린 영역들을 표시
        for i in range(rect_count):

            cv2.rectangle(frame1, rectangles_left[i][0], rectangles_left[i][1], (0, 255, 0), 2)
            cv2.rectangle(frame2, rectangles_right[i][0], rectangles_right[i][1], (0, 255, 0), 2)

        if rect_count == 2 and not drawing:
            cv2.putText(frame1, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

            # 왼쪽 창의 첫 번째와 두 번째로 선택된 영역의 이미지 추출
            img1 = frame1[rectangles_left[0][0][1]:rectangles_left[0][1][1], rectangles_left[0][0][0]:rectangles_left[0][1][0]]
            img3 = frame2[rectangles_left[0][0][1]:rectangles_left[0][1][1], rectangles_left[0][0][0]:rectangles_left[0][1][0]]

            # 이미지 크기를 동일하게 만듦
            img1_resized = cv2.resize(img1, (img3.shape[1], img3.shape[0]))

            # 이미지 그레이스케일로 변환
            img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

            # SSIM 계산
            similarity_index_1, _ = ssim(img1_gray, img3_gray, full=True)
            
        if rect_count == 2 and not drawing:
            # 오른쪽 창의 첫 번째와 두 번째로 선택된 영역의 이미지 추출
            img2 = frame1[rectangles_left[1][0][1]:rectangles_left[1][1][1], rectangles_left[1][0][0]:rectangles_left[1][1][0]]
            img4 = frame2[rectangles_left[1][0][1]:rectangles_left[1][1][1], rectangles_left[1][0][0]:rectangles_left[1][1][0]]

            # 이미지 크기를 동일하게 만듦
            img2_resized = cv2.resize(img2, (img4.shape[1], img4.shape[0]))

            # 이미지 그레이스케일로 변환
            img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

            # SSIM 계산
            similarity_index_2, _ = ssim(img2_gray, img4_gray, full=True)

            if similarity_index_2 > 0.96 and similarity_index_1 > 0.96:
                cv2.putText(frame2, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
            else:
                cv2.putText(frame2, "NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)

            if similarity_index_1 < 0.96:
                # 'FAIL' 텍스트 대신 빨간색 상자로 표시
                cv2.rectangle(frame2, rectangles_right[0][0], rectangles_right[0][1], (0, 0, 255), 2)

            if similarity_index_2 < 0.96:
                cv2.rectangle(frame2, rectangles_right[1][0], rectangles_right[1][1], (0, 0, 255), 2)

            cv2.putText(frame2, f"FPS: {fps:.2f}", (frame2.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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

        # 두 개의 화면을 가로로 합쳐서 하나의 이미지로 만듦
        combined_frame = np.concatenate((frame1, frame2), axis=1)

        # 합쳐진 이미지를 출력
        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)

        # Set image to QLabel
        label.setPixmap(QPixmap.fromImage(image))

        cv2.imshow('Vision Sensor', combined_frame)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break

    # 웹캠 종료
    cap.release()
    cv2.destroyAllWindows()

    plt.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    run_ssim()