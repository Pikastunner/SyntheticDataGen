import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QLabel, QDialog, QFormLayout, QLineEdit
)
from PyQt5.QtGui import QIcon

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(300, 200)
        
        # Create form layout for the settings
        layout = QFormLayout()
        
        # Add settings options (you can expand this)
        self.option1 = QLineEdit(self)
        self.option2 = QLineEdit(self)
        
        layout.addRow("Option 1:", self.option1)
        layout.addRow("Option 2:", self.option2)
        
        # Add a save button
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_settings)
        layout.addRow(save_button)
        
        self.setLayout(layout)

    def save_settings(self):
        # Logic to save settings (can be expanded)
        print(f"Option 1: {self.option1.text()}")
        print(f"Option 2: {self.option2.text()}")
        self.accept()


def settingsButton(parentWidget: QWidget | QMainWindow):
    settings_button = QPushButton(parentWidget)
    settings_icon = QIcon("src/Assets/setting.svg")
    settings_button.setIcon(settings_icon)
    settings_button.setIconSize(settings_button.sizeHint())
    settings_button.setFixedSize(40, 40)
    settings_button.clicked.connect(parentWidget.open_settings_window)



class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main App with Settings")
        self.setFixedSize(700, 650)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QVBoxLayout(self.central_widget)

        # Add a label to simulate different pages
        main_layout.addWidget(QLabel("This is the main page"))
        main_layout.addStretch(1)

        # Add a settings button with a gear icon at the bottom left
        bottom_layout = QHBoxLayout()
        # Create the button with an icon (gear icon)        
        bottom_layout.addWidget(settingsButton(self))
        bottom_layout.addStretch(1)  # Push the button to the left

        # Add the bottom layout with settings button to the main layout
        main_layout.addLayout(bottom_layout)

    def open_settings_window(self):
        # Open the settings window
        self.settings_window = SettingsWindow(self)
        self.settings_window.exec_()


if __name__ == "__main__":
    # Main application setup
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
