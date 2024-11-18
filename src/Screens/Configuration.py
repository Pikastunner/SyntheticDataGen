from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal
from Screens.Loader import LoadingScreen, LoadingWorkerFinishing
from camera import is_camera_connected

# Configuration class for the data storage and number of images
class Configuration(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        title = QLabel("Configure Data Storage & Images")
        title.setObjectName("Label1")

        description = QLabel("Set up your data directory and customize the number of images to download.")
        description.setObjectName("Label2")

        output_directory_title = QLabel("Output Directory")
        output_directory_title.setObjectName("SubHeading")

        output_directory_description = QLabel("Choose a location on your device to store all the generated data.")
        output_directory_description.setObjectName("Label2")
        
        # Directory Saving Section
        directory_saving_area = QWidget(objectName="PreprocessingDirectoryArea")
        directory_saving_area.setContentsMargins(10, 15, 10, 15)
        directory_text_layout = QVBoxLayout(directory_saving_area)
        directory_text_layout.addWidget(QLabel("Select a directory to save the synthetic data.", objectName="PreprocessingDirectoryInstructions"))

        self.directory_input = QLineEdit(objectName="PreprocessingDirectoryInput")
        self.directory_input.setFixedHeight(25)

        browse_button = QPushButton("Browse")
        browse_button.setFixedHeight(25)
        browse_button.clicked.connect(self.select_directory)

        directory_input_layout = QHBoxLayout()
        directory_input_layout.addWidget(self.directory_input)
        directory_input_layout.addWidget(browse_button)
        directory_text_layout.addLayout(directory_input_layout)


        num_images_title = QLabel("Number of Images to Download")
        num_images_title.setObjectName("SubHeading")
        
        num_images_description = QLabel("Specify how many images you'd like to download for processing.")
        num_images_description.setObjectName("Label2")

        num_images_input_section = QWidget()
        num_images_input_section.setObjectName("NumberSpecifySection")

        self.num_images_input = QSpinBox(objectName="NumberImagesInput")
        self.num_images_input.setRange(1, 2147483647)  # 2^31 - 1, a common "maximum" value for practical purposes
        self.num_images_input.setValue(10)  # Default value
        self.num_images_input.setFixedSize(100, 25)
        self.num_images_input.valueChanged.connect(self.change_size_estimate)

        num_images_layout = QHBoxLayout()
        num_images_caption = QLabel("Number of Images:")
        num_images_caption.setObjectName("NumImageCaption")
        num_images_layout.addWidget(num_images_caption, 20)
        num_images_layout.addWidget(self.num_images_input, 30)
        num_images_layout.addWidget(QWidget(), 50)

        num_images_input_section.setLayout(num_images_layout)

        self.size_estimate_text = QLabel("", objectName="Label2")
        self.change_size_estimate()

        text_section = QWidget()
        text_layout = QVBoxLayout()

        text_layout.addWidget(title)
        text_layout.addWidget(description)
        text_layout.addWidget(output_directory_title)
        text_layout.addWidget(output_directory_description)
        text_layout.addWidget(directory_saving_area)
        text_layout.addWidget(num_images_title)
        text_layout.addWidget(num_images_description)
        text_layout.addWidget(num_images_input_section)
        text_layout.addWidget(self.size_estimate_text)

        text_section.setLayout(text_layout)

        empty_area = QWidget()
        empty_area.setObjectName("UploadTextSection")

        # Navigation Section
        navigation_area = QWidget(objectName="PreprocessingNavigationArea")
        navigation_buttons_layout = QHBoxLayout(navigation_area)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        back_button = QPushButton("Back", objectName="BackPageButton")
        back_button.setFixedSize(100, 30)
        back_button.clicked.connect(self.go_to_back_page)

        next_button = QPushButton("Next", objectName="NextPageButton")
        next_button.setFixedSize(100, 30)
        next_button.clicked.connect(self.go_to_next_page)

        navigation_buttons_layout.addWidget(spacer)
        navigation_buttons_layout.addWidget(back_button)
        navigation_buttons_layout.addSpacing(10)
        navigation_buttons_layout.addWidget(next_button)
        navigation_buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(text_section, 70)
        main_layout.addWidget(empty_area, 20)
        main_layout.addWidget(navigation_area, 10)
        self.setLayout(main_layout)
        
    def go_to_back_page(self):
        par = self.parent.stacked_widget
        current_index = par.currentIndex()
        if current_index > 0:
            par.setCurrentIndex(current_index - 1) 
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        if (len(self.directory_input.text()) == 0):
            error_msg = QMessageBox()
            
            # Set critical icon for error message
            error_msg.setIcon(QMessageBox.Critical)
            
            # Set the title of the error message box
            error_msg.setWindowTitle("Empty Path Error")
            
            # Set the detailed text to help the user troubleshoot
            error_msg.setText('<span style="color:#005ace;font-size: 15px;">No Path Input!</span>')
            error_msg.setInformativeText("Please make sure you specify a path")
            
            # Set the standard button to close the message box
            error_msg.setStandardButtons(QMessageBox.Ok)

            # Execute and show the message box
            error_msg.exec_()
            return
        par = self.parent.stacked_widget
        current_index = par.currentIndex()
        if current_index < par.count() - 1:
            t_estimate = 50   # 50 seconds
            self.loading_screen = LoadingScreen(self.parent, t_estimate)
            self.loading_screen.show()
            par.setCurrentIndex(par.currentIndex() + 1)
            self.next_screen = par.widget(par.currentIndex())
            # self.num_images_input.value(), self.directory_input.text()
            self.loading_worker = LoadingWorkerFinishing(
                self.num_images_input.value(), self.directory_input.text(),
                self.next_screen, par
            )
            self.loading_worker.finished_f_signal.connect(self.on_loading_finished)
            self.loading_worker.start()
        else:
            print("Already on the last page")

    def on_loading_finished(self):
        # self.next_screen.update_variables(triangle_mesh, dir_input)
        self.next_screen.setup_gui()
        self.loading_screen.close()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)

    def image_size_estimate(self):
        # Estimated PNG Size ≈ Width × Height × Bytes Per Pixel × Number of Images / 1000 (to convert to KB)
        uncompressed_formula = 640 * 480 * 3 * self.num_images_input.value() / 1000  # Base estimate in KB
        return [int(uncompressed_formula * 0.5), int(uncompressed_formula * 0.7)]  # Return as KB with compression factors

    def change_size_estimate(self):
        def format_size(size_in_kb):
            """Convert size in KB to an appropriate size unit (KB, MB, GB)."""
            if size_in_kb < 1024:
                return f"{size_in_kb} KB"
            elif size_in_kb < 1024**2:
                return f"{size_in_kb / 1024:.2f} MB"
            else:
                return f"{size_in_kb / 1024**2:.2f} GB"
        size_estimate = self.image_size_estimate()
        size_text_min = format_size(size_estimate[0])
        size_text_max = format_size(size_estimate[1])
        self.size_estimate_text.setText(f"Estimated size ≈ {size_text_min} - {size_text_max}")
