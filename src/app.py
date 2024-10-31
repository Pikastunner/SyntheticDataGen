import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget)

from Screens.Finishing import FinishingScreen
from Screens.CapturedPhotoReview import CapturedPhotoReviewScreen
from Screens.Preview import PreviewScreen
from Screens.Preprocessing import PreprocessingScreen
from Screens.Welcome import WelcomeScreen


# Main Application
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Data Generator")
        self.setFixedSize(700, 650)
        self.screens = {}
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Welcome screen and preview screen
        self.welcome_screen = WelcomeScreen(self.central_widget)
        self.preview_screen = PreviewScreen(self.central_widget)
        self.captured_photoReview_screen = CapturedPhotoReviewScreen(self.central_widget)
        self.preprocessingScreen = PreprocessingScreen(self.central_widget)
        # self.finishing_screen = FinishingScreen(self.central_widget)

        # Add screens to the stacked widget
        self.central_widget.addWidget(self.welcome_screen)
        self.central_widget.addWidget(self.preview_screen)
        self.central_widget.addWidget(self.captured_photoReview_screen)
        self.central_widget.addWidget(self.preprocessingScreen)
        # self.central_widget.addWidget(self.finishing_screen)
        
        self.central_widget.setCurrentIndex(0)  # Start with Welcome Screen


import os

if __name__ == '__main__':
    app = QApplication(sys.argv)

    with open("src/style.qss","r") as fh:
        app.setStyleSheet(fh.read())

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
