import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QLabel, QDialog, QFormLayout, QLineEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import json

from Screens.Constants import OUTPUT_PATH, OUT, SETTINGS_FILE, SETTINGS_ICON


class SettingsWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                self.settings_dict = json.load(f)
        else:
            self.settings_dict = {
                "Default Image Save Path": OUTPUT_PATH,
                "Synthetic Data Generation Output Directory": OUT
            }

        self.setWindowTitle("Settings")
        self.setFixedSize(300, 200)
        self.settingsFile = os.path.dirname(os.path.abspath(__file__))
        
        # Create form layout for the settings
        layout = QFormLayout()
        
        # Add settings options (you can expand this)
        self.option1 = QLineEdit(self)
        self.option2 = QLineEdit(self)
        
        layout.addRow("Default Image Save Path:", self.option1)
        layout.addRow("Synthetic Data Generation Output Directory:", self.option2)
        
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


def create_settings_button(parent):
    settings_button = QPushButton(parent)
    settings_icon = QIcon(SETTINGS_ICON)
    settings_button.setIcon(settings_icon)
    settings_button.setIconSize(settings_button.sizeHint())
    settings_button.setFixedSize(40, 40)

    layout = QHBoxLayout()
    layout.addWidget(settings_button, 0, Qt.AlignLeft | Qt.AlignBottom)
    layout.addStretch(1)

    def open_settings_dialog(parent):
        dialog = SettingsWindow(parent)
        dialog.exec_()  # Open the settings window as a modal dialog

    settings_button.clicked.connect(lambda: open_settings_dialog(parent))

    return settings_button, layout


# Main screen that uses the reusable settings button
class MainScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Screen with Settings")
        self.setFixedSize(500, 400)
        
        main_layout = QVBoxLayout()

        label = QLabel("Welcome to the Main Screen")
        label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(label)

        main_layout.addStretch(1)

        settings_button, settings_layout = create_settings_button(self)
        
        main_layout.addLayout(settings_layout)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainScreen()
    main_window.show()
    sys.exit(app.exec_())