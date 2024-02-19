# qr, barcode

import cv2
from pyzbar.pyzbar import decode
from PySide2.QtGui import QImage, QPixmap
import time

def read_barcodes(frame):
    barcodes = decode(frame)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        barcode_info = barcode.data.decode('utf-8')
        print(f"Detected barcode or QR code: {barcode_info}")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 0.5, (255, 0, 0), 1)

    return frame

def run_qr_code(cap, label):
    
    # 초당 프레임 수를 측정하기 위한 변수
    start_time = time.time()
    frame_count = 0
    fps = 0  # 초기화된 fps 변수 추가
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


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
            
        frame = read_barcodes(frame)

        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)

        # Set image to QLabel
        label.setPixmap(QPixmap.fromImage(image))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()