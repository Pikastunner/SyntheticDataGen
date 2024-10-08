import sys
import os
import re
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QSpacerItem,
    QSizePolicy, QFrame, QStackedWidget, QMainWindow
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
OUTPUT_PATH = "../test_images"
# OUTPUT_PATH = "input_images"

# Component 5
class CapturedPhotoReviewScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.resize(900, 600)
        sortKey = lambda s: int(re.match(r".*_(\d+).png", s).group(1))
        isRGB = lambda p: p.startswith("rgb_ima")
        fnames = sorted(filter(isRGB, os.listdir(OUTPUT_PATH)), key=sortKey)
        self.img_paths = [f"{OUTPUT_PATH}/{f}" for f in fnames]
        self.main_img_path = self.img_paths[0] if self.img_paths else None
        self.main_img_frame = self.image_frame(self.main_img_path, 640, 480)
        self.sub_img_frame1 = self.image_frame(None, 220, 165)
        self.sub_img_frame2 = self.image_frame(None, 220, 165)
        self.sub_img_frame3 = self.image_frame(None, 220, 165)

        # Set up the main layout
        main_layout = QVBoxLayout()

        # Row 1: Heading "Captured Photo Review"
        heading_label = QLabel("Captured Photo Review")
        heading_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        heading_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(heading_label)

        # Row 2: Text "Confirm that these are the photos to use"
        confirm_label = QLabel("Confirm that these are the photos to use")
        main_layout.addWidget(confirm_label)

        # Row 3: Two columns with a large image frame and three smaller image frames
        row3_layout = QHBoxLayout()

        # Column 1: Large image frame
        main_pic_layout = QVBoxLayout()
        main_pic_heading = QLabel("Selected Image")
        main_pic_heading.setAlignment(Qt.AlignCenter)
        main_pic_layout.addWidget(main_pic_heading)
        main_pic_layout.addWidget(self.main_img_frame)

        main_pic_buttons = QHBoxLayout()
        left_button = QPushButton()
        delete_button = QPushButton("Delete")
        right_button = QPushButton()
        left_button.setIcon(self.style().standardIcon(self.style().SP_ArrowLeft))
        right_button.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
        left_button.clicked.connect(self.prev_img)
        delete_button.clicked.connect(self.delete_img)
        right_button.clicked.connect(self.next_img)

        main_pic_buttons.addWidget(left_button)
        main_pic_buttons.addWidget(delete_button)
        main_pic_buttons.addWidget(right_button)

        main_pic_layout.addLayout(main_pic_buttons)
        row3_layout.addLayout(main_pic_layout)
        row3_layout.setAlignment(main_pic_layout, Qt.AlignCenter)

        # Column 2: Three smaller image frames
        small_images_layout = QVBoxLayout()
        queued_pics_label = QLabel("Queued Images")
        queued_pics_label.setAlignment(Qt.AlignCenter)
        small_images_layout.addWidget(queued_pics_label)

        small_images_layout.addWidget(self.sub_img_frame1)
        small_images_layout.addWidget(self.sub_img_frame2)
        small_images_layout.addWidget(self.sub_img_frame3)

        spacerLayout = QHBoxLayout()
        spacerLayout.addItem(QSpacerItem(
            22, left_button.sizeHint().height(),
            QSizePolicy.Minimum, QSizePolicy.Expanding
        ))
        small_images_layout.addLayout(spacerLayout)

        row3_layout.addLayout(small_images_layout)
        main_layout.addLayout(row3_layout)

        # Row 4: Buttons "Back" and "Next", right aligned
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Pushes the buttons to the right

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.go_to_back_page)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.go_to_next_page)

        button_layout.addWidget(back_button)
        button_layout.addWidget(next_button)

        main_layout.addLayout(button_layout)

        # Set the layout to the main window
        self.setLayout(main_layout)
        self.setWindowTitle('Photo Review App')

        self.display_images()


    def newPixmap(self, frame: QLabel, img_path: str) -> QPixmap:
        if not img_path:
            frame.clear()
            frame.setPixmap(QPixmap())
        pmap = QPixmap(img_path).scaled(
            frame.width(),
            frame.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        return pmap

    def image_frame(self, img_path: str, width: int, length: int) -> QLabel:
        frame = QLabel(self)
        frame.setFixedSize(width, length)
        frame.setStyleSheet("border: 2px solid black;")
        if img_path:
            frame.setPixmap(self.newPixmap(frame, img_path))
        return frame

    def display_images(self):
        if newPmap:= self.newPixmap(self.main_img_frame, self.main_img_path):
            self.main_img_frame.setPixmap(newPmap)
        if not self.main_img_path:
            return
        i = self.img_paths.index(self.main_img_path)
        sub_image = lambda x: self.img_paths[x] if x < len(self.img_paths) else None
        if newPmap:= self.newPixmap(self.sub_img_frame1, sub_image(i + 1)):
            self.sub_img_frame1.setPixmap(newPmap)
        if newPmap:= self.newPixmap(self.sub_img_frame2, sub_image(i + 2)):
            self.sub_img_frame2.setPixmap(newPmap)
        if newPmap:= self.newPixmap(self.sub_img_frame3, sub_image(i + 3)):
            self.sub_img_frame3.setPixmap(newPmap)
        
    def delete_img(self):
        if not self.main_img_path:
            return
        i = self.img_paths.index(self.main_img_path)
        self.img_paths.remove(self.main_img_path)
        if not self.img_paths:
            self.main_img_path = None
        elif i > len(self.img_paths) - 1:
            self.main_img_path = self.img_paths[i - 1]
        else:
            self.main_img_path = self.img_paths[i]
        self.display_images()

    def prev_img(self):
        i = self.img_paths.index(self.main_img_path)
        if i == 0:
            return
        self.main_img_path = self.img_paths[i - 1]
        self.display_images()

    def next_img(self):
        i = self.img_paths.index(self.main_img_path)
        if i == len(self.img_paths) - 1:
            return
        self.main_img_path = self.img_paths[i + 1]
        self.display_images()

    def prev_page(self):
        self.parent.setCurrentIndex(1)


    def go_to_back_page(self):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 1) 
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
        else:
            print("Already on the last page")

if __name__ == "__main__":
    # Run the CapturedPhotoReviewScreen on its own
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("Photo Review Application")
            self.setGeometry(100, 100, 400, 400)

            # Stacked widget to manage the multiple pages
            self.stacked_widget = QStackedWidget()
            self.setCentralWidget(self.stacked_widget)

            # Initialize pages
            self.photo_review_page = CapturedPhotoReviewScreen(self)
            self.stacked_widget.addWidget(self.photo_review_page)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
