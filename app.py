import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from detection.detector import FaceDetector
from preprocessing.preprocessor import Preprocessor
from recognition.fisher import FisherRecognizer
from common import utils
from skimage import io

appdata = "appdata"
names_file = os.path.join(appdata, "names.txt")
recognizer_file = os.path.join(appdata, "recognizer.pkl")

if not os.path.exists(appdata):
    os.mkdir(appdata)

if not os.path.exists(names_file):
    with open(names_file, "w") as file:
        file.write("")


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.opened_images: list[str] = []
        self.people: list[str]

        self.faces_changed = False
        self.detector = FaceDetector()
        self.preprocessor = Preprocessor()
        self.recognizer = FisherRecognizer()
        self.detector.load()
        if os.path.exists(recognizer_file):
            self.recognizer.load(recognizer_file)

        self.load_names()

        # Create a layout to hold the widgets
        layout = QtWidgets.QVBoxLayout(self)

        # Create a scroll area to display the images
        self.scroll_area_main_images = QtWidgets.QScrollArea(self)
        self.scroll_area_main_images.setWidgetResizable(True)
        self.scroll_area_main_images.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )

        # Create a frame to display the images
        self.image_frame = QtWidgets.QFrame(self)
        self.image_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Create a horizontal layout for the images
        self.image_layout = QtWidgets.QHBoxLayout(self.image_frame)

        # Set the scroll area to display the frame
        self.scroll_area_main_images.setWidget(self.image_frame)
        layout.addWidget(self.scroll_area_main_images)

        # Create a horizontal layout for the buttons
        button_layout_1 = QtWidgets.QHBoxLayout()

        # Create a button to add images
        self.add_button = QtWidgets.QPushButton("Add Images", self)
        self.add_button.clicked.connect(self.add_images)
        button_layout_1.addWidget(self.add_button)

        # Create a button to clear all images
        self.clear_button = QtWidgets.QPushButton("Clear All", self)
        self.clear_button.clicked.connect(self.clear_image_layout)
        button_layout_1.addWidget(self.clear_button)

        # Create a horizontal layout for the buttons
        button_layout_2 = QtWidgets.QHBoxLayout()

        # Create a button to manage the people list
        self.manage_saved_people_button = QtWidgets.QPushButton(
            "Manage Saved People", self
        )
        self.manage_saved_people_button.clicked.connect(self.manage_saved_people)
        button_layout_2.addWidget(self.manage_saved_people_button)

        # Create a button to generate the output
        self.generate_button = QtWidgets.QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.generate)
        button_layout_2.addWidget(self.generate_button)

        # Add the button layout to the main layout
        layout.addLayout(button_layout_1)
        layout.addLayout(button_layout_2)

        # Set the main window properties
        self.setWindowTitle("RecognEYEze!")
        self.setGeometry(100, 100, 800, 600)

    def add_images(self):
        # options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.ReadOnly
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)",
            # options=options,
        )
        for file_name in file_names:
            # Load the image and display it in a label
            image = QtGui.QPixmap(file_name)
            image = image.scaled(300, 300, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

            # Create a label to display the image
            label = QtWidgets.QLabel(self)
            label.setPixmap(image)
            label.setFixedSize(300, 300)

            # Add the label to the layout
            self.image_layout.addWidget(label)
            self.opened_images.append(file_name)

    def clear_image_layout(self):
        # Clear the list of opened images
        self.opened_images = []

        # Remove all widgets from the image layout
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            self.image_layout.removeWidget(widget)

    def manage_saved_people(self):
        # Create a new window to display the list of names
        window = QtWidgets.QDialog(self)
        window.setWindowTitle("Manage Saved People")
        window.setGeometry(150, 150, 600, 600)

        self.name_layout = QtWidgets.QVBoxLayout(window)

        self.scroll_area_name_layout = QtWidgets.QScrollArea(self)
        self.scroll_area_name_layout.setWidgetResizable(True)

        # _areaCreate a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Create an add button
        add_button = QtWidgets.QPushButton("Add New Person", self)
        add_button.clicked.connect(self.add_name)
        button_layout.addWidget(add_button)

        # Create a done button
        done = QtWidgets.QPushButton("Done", self)
        done.clicked.connect(window.close)
        done.clicked.connect(self.train)
        button_layout.addWidget(done)

        # Add the button layout to the main layout
        self.name_layout.addLayout(button_layout)

        self.name_layout.addWidget(self.scroll_area_name_layout)

        self.gen_name_layout()

        # Show the new window
        window.show()

    def gen_name_layout(self):
        self.name_layout.removeWidget(self.scroll_area_name_layout)

        # Create a scroll area to display the list of names
        self.scroll_area_name_layout = QtWidgets.QScrollArea(self)
        self.scroll_area_name_layout.setWidgetResizable(True)

        # Create a frame to hold the list of names
        frame = QtWidgets.QFrame(self)

        # Create a vertical layout for the names
        name_layout = QtWidgets.QVBoxLayout(frame)

        # Add a widget for each name
        for name in self.people:
            widget = QtWidgets.QWidget(self)
            widget_layout = QtWidgets.QHBoxLayout(widget)

            # Add a label for the name
            label = QtWidgets.QLabel(name, self)
            widget_layout.addWidget(label)

            # Add a delete button for the name
            delete_button = QtWidgets.QPushButton("Delete", self)
            delete_button.clicked.connect(self.delete_name)
            widget_layout.addWidget(delete_button)

            # Add a change name button for the name
            change_name_button = QtWidgets.QPushButton("Change Name", self)
            change_name_button.clicked.connect(self.change_name)
            widget_layout.addWidget(change_name_button)

            # Add a modifu images button for the name
            modify_images_button = QtWidgets.QPushButton("Modify Images", self)
            modify_images_button.clicked.connect(self.modify_images)
            widget_layout.addWidget(modify_images_button)

            # Add the widget to the layout
            name_layout.addWidget(widget)

        # Set the scroll area to display the frame
        self.scroll_area_name_layout.setWidget(frame)

        self.name_layout.addWidget(self.scroll_area_name_layout)

    def change_name(self):
        sender = self.sender()

        # get name from sender
        c = sender.parent().children()

        name = ""
        for child in c:
            if isinstance(child, QtWidgets.QLabel):
                name = child.text()

        # Create a new window to enter the new name
        window = QtWidgets.QDialog(self)
        window.setWindowTitle("Change Name")
        window.setGeometry(200, 200, 300, 100)

        # Create a layout for the input and buttons
        layout = QtWidgets.QVBoxLayout(window)

        # Create a line edit for the new name
        name_edit = QtWidgets.QLineEdit(self)
        name_edit.setText(name)
        layout.addWidget(name_edit)

        # Create a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Create a cancel button
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        cancel_button.clicked.connect(window.close)
        button_layout.addWidget(cancel_button)

        # Create an ok button
        ok_button = QtWidgets.QPushButton("OK", self)
        ok_button.clicked.connect(lambda: self.update_name(name, name_edit.text()))
        ok_button.clicked.connect(window.close)
        button_layout.addWidget(ok_button)

        # Add the button layout to the main layout
        layout.addLayout(button_layout)

        # Show the new window
        window.show()

    def add_name(self):
        # Create a new window to enter the name
        window = QtWidgets.QDialog(self)
        window.setWindowTitle("Add Name")
        window.setGeometry(200, 200, 300, 100)

        # Create a layout for the input and buttons
        layout = QtWidgets.QVBoxLayout(window)

        # Create a line edit for the name
        name_edit = QtWidgets.QLineEdit(self)
        layout.addWidget(name_edit)

        # Create a horizontal layout for the buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Create a cancel button
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        cancel_button.clicked.connect(window.close)
        button_layout.addWidget(cancel_button)

        # Create an ok button
        ok_button = QtWidgets.QPushButton("OK", self)
        ok_button.clicked.connect(lambda: self.add_new_name(name_edit.text()))
        ok_button.clicked.connect(window.close)
        button_layout.addWidget(ok_button)

        # Add the button layout to the main layout
        layout.addLayout(button_layout)

        # Show the new window
        window.show()

    def delete_name(self):
        sender = self.sender()

        # get name from sender
        c = sender.parent().children()

        name = ""
        for child in c:
            if isinstance(child, QtWidgets.QLabel):
                name = child.text()

        # Remove the name from the list
        self.people.remove(name)

        # Delete the folder
        path = os.path.join(appdata, name)
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)

        self.gen_name_layout()
        self.save_names()

    def update_name(self, old_name, new_name):
        # Remove the old name from the list
        self.people.remove(old_name)

        # Add the new name to the list
        self.people.append(new_name)

        # Rename the folder
        path = os.path.join(appdata, old_name)
        if os.path.exists(path):
            os.rename(os.path.join(appdata, old_name), os.path.join(appdata, new_name))

        self.gen_name_layout()
        self.save_names()

    def add_new_name(self, name):
        # Add the name to the list
        self.people.append(name)

        self.gen_name_layout()
        self.save_names()

    def save_names(self):
        # Save the list of names to a file
        with open(names_file, "w") as f:
            for name in self.people:
                f.write(name + "\n")

    def load_names(self):
        # Load the list of names from a file
        self.people = []
        with open(names_file, "r") as f:
            for line in f:
                self.people.append(line.strip())

    def modify_images(self):
        sender = self.sender()

        # get name from sender
        c = sender.parent().children()

        name = ""
        for child in c:
            if isinstance(child, QtWidgets.QLabel):
                name = child.text()

        # Create a new window to display the images
        window = QtWidgets.QDialog(self)
        window.setWindowTitle(f"Modify {name} Images")
        window.setGeometry(150, 150, 600, 600)

        # Create a layout to hold the widgets
        layout = QtWidgets.QVBoxLayout(window)

        # Create a scroll area to display the images
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )

        # Create a frame to display the images
        image_frame = QtWidgets.QFrame(self)
        image_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Create a layout to hold the images
        image_layout = QtWidgets.QHBoxLayout(image_frame)

        # Load images from appdata/name
        path = os.path.join(appdata, name)
        if os.path.exists(path):
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                # Load the image and display it in a label
                image = QtGui.QPixmap(file_path)
                image = image.scaled(
                    200, 200, QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )

                # Create a label to display the image
                label = QtWidgets.QLabel(self)
                label.setPixmap(image)
                label.setFixedSize(200, 200)

                # Add the label to the layout
                image_layout.addWidget(label)

        scroll_area.setWidget(image_frame)
        layout.addWidget(scroll_area)

        button_layout = QtWidgets.QHBoxLayout()

        add_button = QtWidgets.QPushButton("Add Images", self)
        add_button.clicked.connect(lambda: self.add_person_image(name, image_layout))
        button_layout.addWidget(add_button)

        clear_button = QtWidgets.QPushButton("Clear Images", self)
        clear_button.clicked.connect(
            lambda: self.clear_person_images(name, image_layout)
        )
        button_layout.addWidget(clear_button)

        # Create a done button
        done = QtWidgets.QPushButton("Done", self)
        done.clicked.connect(window.close)
        button_layout.addWidget(done)

        layout.addLayout(button_layout)

        window.show()

    def add_person_image(self, name, image_layout):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)",
            # options=options,
        )
        for file_name in file_names:
            # Load the image
            image = io.imread(file_name)

            # if image is x pixels, a face is expected to be x/3 pixels
            # for image of size 37*3, we want scale to be 1
            # generally, we want scale to be image.shape[0] / 111
            # s = image_float.shape[0] / 111
            # scales = [0.05*s, 0.1*s, 0.2*s, 0.4*s, 0.8*s, s, 1.2*s, 1.4*s, 1.8*s, 2.0*s, 4.0*s]

            # Detect faces in the image
            face, colored = utils.detect_face(image)
            if face is None:
                continue

            # save the face to appdata/name
            path = os.path.join(appdata, name)
            if not os.path.exists(path):
                os.mkdir(path)

            file_name = os.path.join(path, os.path.basename(file_name))
            io.imsave(file_name, colored)

            # Load the image and display it in a label
            image = QtGui.QPixmap(file_name)
            image = image.scaled(200, 200, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

            # Create a label to display the image
            label = QtWidgets.QLabel(self)
            label.setPixmap(image)
            label.setFixedSize(200, 200)

            # Add the label to the layout
            image_layout.addWidget(label)
            self.opened_images.append(file_name)

            self.faces_changed = True

    def clear_person_images(self, name, image_layout):
        path = os.path.join(appdata, name)
        if os.path.exists(path):
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                os.remove(file_path)

        # Remove all the images from the layout
        for i in reversed(range(image_layout.count())):
            widget = image_layout.itemAt(i).widget()
            image_layout.removeWidget(widget)

        self.faces_changed = True

    def train(self):
        if not self.faces_changed:
            return

        faces = []
        labels = []

        for name in os.listdir(appdata):
            path = os.path.join(appdata, name)
            if os.path.isdir(path):
                for file_name in os.listdir(path):
                    file_path = os.path.join(path, file_name)
                    image = io.imread(file_path)
                    face, _ = utils.detect_face(image)
                    if face is None:
                        continue
                    faces.append(face)
                    labels.append(name)

        # Preprocessing
        faces = self.preprocessor.preprocess(faces)

        # Train the model
        self.recognizer.fit(faces, labels)
        self.recognizer.dump(recognizer_file)

    def generate(self):
        # Create a new window to display the list of names
        window = QtWidgets.QDialog(self)
        window.setWindowTitle("List of Names")
        window.setGeometry(150, 150, 400, 600)

        # Create a layout to hold the text box
        layout = QtWidgets.QVBoxLayout(window)

        # Generate the list of names
        faces = []
        for image in self.opened_images:
            face, _ = utils.detect_face(io.imread(image))
            if face is None:
                continue
            faces.append(face)

        faces = self.preprocessor.preprocess(faces)
        names = np.unique(self.recognizer.predict(faces))

        # Create a text box to display the list of names
        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setText("\n".join(names))
        layout.addWidget(self.text_box)

        # Show the new window
        window.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
