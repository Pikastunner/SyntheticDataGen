from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QWidget, QLabel, QDialog, QMainWindow, QApplication
from PyQt5.QtGui import QMovie

from Screens.Constants import WIN_WIDTH, WIN_HEIGHT

import json
import random

class LoadingWorkerPreprocessing(QThread):
    def __init__(self, img_paths, depth_paths, parent=None):
        super().__init__(parent)
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.parent = parent

    def run(self):
        self.parent.setCurrentIndex(self.parent.currentIndex() + 1)
        next_screen = self.parent.widget(self.parent.currentIndex())
        next_screen.update_variables(self.img_paths, self.depth_paths)


class LoadingWorkerFinishing(QThread):
    def __init__(self, triangle_mesh, dir_input: str, parent=None):
        super().__init__(parent)
        self.triangle_mesh = triangle_mesh
        self.dir_input = dir_input
        self.parent = parent

    def run(self):
        # self.msleep(15000)
        self.parent.setCurrentIndex(self.parent.currentIndex() + 1)
        next_screen = self.parent.widget(self.parent.currentIndex())
        next_screen.update_variables(self.triangle_mesh, self.dir_input)


class LoadingScreen(QDialog):
    def __init__(self, parent: QMainWindow | None = None, time_estimate: float = 10):
        super().__init__(parent)

        with open("src/Icons/random_facts.json") as f:
            self.random_facts = json.load(f)

        self.setWindowTitle("Bro, I'm processing. Just chillax!")
        self.setFixedSize(700, 630)
        # self.setFixedSize(700, 650)
        self.setStyleSheet("border-radius: 20px;")
        
        is_light = parent.is_light() if parent else True

        mode = "_light" if is_light else "_dark"
        gPath = f'src/Icons/app_load_animation{mode}.gif'

        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)

        self.gif_label = QLabel(self)
        self.gif_label.setFixedSize(300, 300)
        self.gif_label.setAlignment(Qt.AlignCenter)
        
        # self.movie = QMovie("src/Icons/app_load_animation.gif")
        self.movie = QMovie(gPath)
        self.movie.setScaledSize(self.gif_label.size())
        self.gif_label.setMovie(self.movie)
        
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(self.gif_label, alignment=Qt.AlignCenter)
        layout.addStretch()
        
        self.countdown_label = QLabel(f"Estimated time: {round(time_estimate)} seconds", self)
        colour = "black" if is_light else "white"
        self.countdown_label.setStyleSheet(f"color: {colour}; font-size: 12pt;")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.countdown_label)
        
        self.movie.start()
        
        self.countdown_value = round(time_estimate)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        """Update the countdown text."""
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.countdown_label.setText(f"Estimated time: {self.countdown_value} seconds")
        else:
            if self.countdown_value == 0:
                self.countdown_label.setText(f"Fun Fact: {random.choice(self.random_facts)}")
            elif self.countdown_value == -4:
                self.countdown_value = 1

    def showEvent(self, event):
        # Center the loading screen over the parent window when it's shown
        if self.parent():
            parent_rect = self.parent().geometry()
            self.move(
                parent_rect.center().x() - self.width() // 2,
                parent_rect.center().y() - self.height() // 2
            )
            center_point = parent_rect.center()
            dialog_geometry = self.frameGeometry()
            dialog_geometry.moveCenter(center_point)
            self.move(dialog_geometry.topLeft())

        super().showEvent(event)
