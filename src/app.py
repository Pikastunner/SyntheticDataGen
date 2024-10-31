import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize

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
        
        # Central widget for layout management
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout that will contain the stacked widget
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the layout
        self.main_layout.setSpacing(0)  # Remove spacing between elements in the layout
        
        # Main widget for switching between scenes
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)  # Ensure no margins inside the stacked widget
        self.main_layout.addWidget(self.stacked_widget)
        
        # Add screens
        self.stacked_widget.addWidget(WelcomeScreen(self.stacked_widget))
        self.stacked_widget.addWidget(PreviewScreen(self.stacked_widget))
        self.stacked_widget.addWidget(CapturedPhotoReviewScreen(self.stacked_widget))
        self.stacked_widget.addWidget(PreprocessingScreen(self.stacked_widget))
        
        self.stacked_widget.setCurrentIndex(0)  # Start with the Welcome screen

        # Add settings button to each of the Screens
        self.light = True

        self.create_light_button()

    ## CREATION/LOGIC OF LIGHT/DARK MODE SWITCH
    def create_light_button(self):        
        # Settings button with custom SVG icon
        self.light_switch = QPushButton(self.central_widget, objectName="LightButton")
        self.light_switch.setIcon(QIcon("src/Icons/sun.svg"))
        self.light_switch.setIconSize(QSize(20, 20))
        self.light_switch.setFixedSize(30, 30)
        self.light_switch.setToolTip("Settings")

        # Align the button to the top-right corner with a slight offset
        self.light_switch.move(self.width() - self.light_switch.width() - 10, 10)  # Adjust offset as needed
        
        # Update position if the window is resized
        self.resizeEvent = self.update_light_button_position

        # Connect the button to open settings
        self.light_switch.clicked.connect(self.toggle_light)

    def update_light_button_position(self, event):
        """Update settings button position on window resize."""
        self.light_switch.move(self.width() - self.light_switch.width() - 10, 10)
        super().resizeEvent(event)


    def toggle_light(self):
        # Placeholder for the settings functionality
        print("Light button clicked")

        stylesheet = "src/Stylesheets/style_dark.qss" if self.light else "src/Stylesheets/style_light.qss"
        icon = "src/Icons/moon.svg" if self.light else "src/Icons/sun.svg"
        self.light = not self.light

        with open(stylesheet, "r") as fh:
            app.setStyleSheet(fh.read())
            
        self.light_switch.setIcon(QIcon(icon))



if __name__ == '__main__':
    app = QApplication(sys.argv)

    with open("src/Stylesheets/style_light.qss", "r") as fh:
        app.setStyleSheet(fh.read())

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())