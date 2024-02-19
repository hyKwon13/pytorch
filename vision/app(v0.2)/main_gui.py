# v0.2

import sys
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout
import cv2
import qr_code, yolo, ocr, ssim, a1_contrast, a2_edge, a3_position, a4_contour, a5_pitch, a6_width, a7_diameter # Rpi에서는 from function2 import qr ...

class VideoProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.label = QLabel(self)
        self.label.setScaledContents(True)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Vision Sensor')
        layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        # 버튼 생성 및 참조 저장
        self.buttons = {
            'qr': QPushButton('QR Code'),
            'yolo': QPushButton('Blob Count'),
            'ocr': QPushButton('OCR'),
            'ssim': QPushButton('돌출 검출'),
            'a1_contrast': QPushButton('대조'),
            'a2_edge': QPushButton('에지 유무'),
            'a3_position': QPushButton('위치 보정'),
            'a4_contour': QPushButton('윤곽'),
            'a5_pitch': QPushButton('피치 검출'),
            'a6_width': QPushButton('폭'),
            'a7_diameter': QPushButton('직경'),
        }

        # 버튼에 기능 연결 및 레이아웃에 추가
        for key, button in self.buttons.items():
            button.clicked.connect(getattr(self, f'run_{key}'))
            button_layout.addWidget(button)
            button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    font-size: 16pt;
                    height: 30px;
                }
            """)

        layout.addLayout(button_layout)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setSizePolicy(size_policy)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def reset_button_styles(self):
        for button in self.buttons.values():
            button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    font-size: 16pt;
                    height: 30px;
                }
            """)

    def change_button_style(self, button_key):
        self.buttons[button_key].setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 16pt; height: 30px;")

    def run_qr(self):
        self.reset_button_styles()
        self.change_button_style('qr')
        qr_code.run_qr_code(self.cap, self.label)

    def run_yolo(self):
        self.reset_button_styles()
        self.change_button_style('yolo')
        yolo.run_yolo(self.cap, self.label)

    def run_ocr(self):
        self.reset_button_styles()
        self.change_button_style('ocr')
        ocr.run_ocr(self.cap, self.label)

    def run_ssim(self):
        self.reset_button_styles()
        self.change_button_style('ssim')
        ssim.run_ssim(self.cap, self.label)
        
    def run_a1_contrast(self):
        self.reset_button_styles()
        self.change_button_style('a1_contrast')
        a1_contrast.run_a1_contrast(self.cap, self.label)
        
    def run_a2_edge(self):
        self.reset_button_styles()
        self.change_button_style('a2_edge')
        a2_edge.run_a2_edge(self.cap, self.label)
        
    def run_a3_position(self):
        self.reset_button_styles()
        self.change_button_style('a3_position')
        a3_position.run_a3_position(self.cap, self.label)
        
    def run_a4_contour(self):
        self.reset_button_styles()
        self.change_button_style('a4_contour')
        a4_contour.run_a4_contour(self.cap, self.label)

    def run_a5_pitch(self):
        self.reset_button_styles()
        self.change_button_style('a5_pitch')
        a5_pitch.run_a5_pitch(self.cap, self.label)
        
    def run_a6_width(self):
        self.reset_button_styles()
        self.change_button_style('a6_width')
        a6_width.run_a6_width(self.cap, self.label)

    def run_a7_diameter(self):
        self.reset_button_styles()
        self.change_button_style('a7_diameter')
        a7_diameter.run_a7_diameter(self.cap, self.label)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()
        
def main():
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()