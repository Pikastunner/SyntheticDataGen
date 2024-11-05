from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QWidget, QLabel, QDialog, QMainWindow, QApplication

from Screens.Constants import KPROCESS, WIN_WIDTH, WIN_HEIGHT

class LoadingWorker(QThread):
    progress_changed = pyqtSignal(int)

    def __init__(self, img_paths, depth_paths, parent=None):
        super().__init__(parent)
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.parent = parent

    def run(self):
        self.parent.setCurrentIndex(self.parent.currentIndex() + 1)
        next_screen = self.parent.widget(self.parent.currentIndex())
        next_screen.update_variables(self.img_paths, self.depth_paths, self.progress_changed)
        self.progress_changed.emit(100)


class LoadingScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set the dialog to be modal so it stays on top of the main window
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("Loading...")

        # Remove the window title bar for a simpler look
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)

        # Create and set up the progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Loading, please wait..."))
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        # Set a fixed size for the dialog
        self.setWindowTitle("Progress Screen")
        self.setFixedSize(WIN_WIDTH, WIN_HEIGHT)

    def showEvent(self, event):
        # Center the loading screen over the parent window when it's shown
        if self.parent():
            parent_rect = QApplication.desktop().screenGeometry
            self.move(
                parent_rect.center().x() - self.width() // 2,
                parent_rect.center().y() - self.height() // 2
            )
            center_point = parent_rect.center()
            dialog_geometry = self.frameGeometry()
            dialog_geometry.moveCenter(center_point)
            self.move(dialog_geometry.topLeft())

        super().showEvent(event)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
