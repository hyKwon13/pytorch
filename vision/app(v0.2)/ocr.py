# OCR

import cv2
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import time
from PySide2.QtGui import QImage, QPixmap

def run_ocr(cap, label):
    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, top_left_pt, bottom_right_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            top_left_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bottom_right_pt = (x, y)

    def perform_ocr(frame, top_left_pt, bottom_right_pt):
        nonlocal first_rectangle_drawn, first_ocr_value
        if top_left_pt == (-1, -1) or bottom_right_pt == (-1, -1):
            return
        roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
        if roi.size == 0:
            return
        ocr = PaddleOCR(det_model_dir='./model/ocr/det_onnx2/model.onnx',
                        rec_model_dir='./model/ocr/rec_onnx2/model.onnx',
                        use_onnx=True,
                        use_angle_cls=False,
                        lang='en',
                        show_log=False)
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        result = ocr.ocr(roi, cls=False)
        result = result[0]
        if result is not None:
            for line in result:
                print(line[1][0])
            cv2.putText(frame, line[1][0], (top_left_pt[0], bottom_right_pt[1]+ 21), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if not first_rectangle_drawn:
                cv2.rectangle(frame0, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
                cv2.putText(frame0, line[1][0], (top_left_pt[0], bottom_right_pt[1]+ 21), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2, cv2.LINE_AA)
                first_ocr_value = ''.join(filter(lambda x: x not in [' '], line[1][0]))
                first_rectangle_drawn = True
            else:
                current_ocr_value = ''.join(filter(lambda x: x not in [' '], line[1][0]))
                if current_ocr_value == first_ocr_value:
                    cv2.putText(frame, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                else:
                    cv2.putText(frame, "NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)

    cv2.namedWindow('Vision Sensor')
    cv2.setMouseCallback('Vision Sensor', draw_rectangle)
    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    first_rectangle_drawn = False
    first_ocr_value = ""  # 첫 프레임에서의 OCR 값 저장을 위한 변수

    # 초당 프레임 수 측정을 위한 초기화
    start_time = time.time()
    frame_count = 0
    fps = 0

    for _ in range(30):
        _, _ = cap.read()

    ret, frame0 = cap.read()
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = current_time
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
            perform_ocr(frame, top_left_pt, bottom_right_pt)
            cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
        combined_frame = np.concatenate((frame0, frame), axis=1)
        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
        
        cv2.imshow('Vision Sensor', combined_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    run_ocr()