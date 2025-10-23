# Design a small interface using the core functionality of the PyQt5 system. Add an image to the window in this interface.

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Display Example')
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Create a label to hold the image
        self.image_label = QLabel(self)
        pixmap = QPixmap('/Users/omen/Downloads/archive/train/images/0a9999a432bd9e50.jpg')  # Replace with your image path
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWindow()
    sys.exit(app.exec_())

