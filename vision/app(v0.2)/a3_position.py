# 위치 보정

from ultralytics import YOLO
import cv2
import math
from PySide2.QtGui import QImage, QPixmap
import time
import numpy as np

def run_a3_position(cap, label):
    # start webcam

    # model
    model = YOLO("model/witch.onnx")

    # object classes
    classNames = ["OK", "NG"]

    # Dictionary to store the count of each object class
    class_counts = {class_name: 0 for class_name in classNames}

    # 초당 프레임 수를 측정하기 위한 변수
    start_time = time.time()
    frame_count = 0
    fps = 0  # 초기화된 fps 변수 추가
    first_rectangle_drawn = False
    count_text = ''
    for _ in range(30):  # 예시로 처음 30프레임을 무시하도록 설정 (적절한 수를 선택)
        _, _ = cap.read()

    ret, frame0 = cap.read()
    
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # Reset class counts for the current frame
        frame_class_counts = {class_name: 0 for class_name in classNames}

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
            
        cv2.putText(img, f"FPS: {fps:.2f}", (img.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not first_rectangle_drawn:
            if '1' in count_text:
                cv2.putText(frame0, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)

                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # put box in cam
                        cv2.rectangle(frame0, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # class name
                        cls = int(box.cls[0])
                        class_name = classNames[cls]

                        # Increment the class-specific counter for the current frame
                first_rectangle_drawn = True  # 첫 번째 사각형이 그려졌다고 표시
                      
        # coordinates
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                class_name = classNames[cls]
                print("Class name -->", class_name)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, class_name, org, font, fontScale, color, thickness)

                # Increment the class-specific counter for the current frame
                frame_class_counts[class_name] += 1
                
        # Update the total count for each class
        for class_name, count in frame_class_counts.items():
            class_counts[class_name] = count

        # Display the count of each class for the current frame
        count_text = ''
        for class_name, count in class_counts.items():
            count_text += f'{class_name}: {count}   '


        if frame_class_counts['OK'] >= 1 and frame_class_counts['NG'] == 0:
            cv2.putText(img, "OK", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
        elif frame_class_counts['NG'] >= 1:
            cv2.putText(img, "NG", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)


        combined_frame = np.concatenate((frame0, img), axis=1)

        frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to QImage
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
                
        cv2.imshow('Vision Sensor', combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_a3_position()