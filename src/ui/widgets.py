import time
import pickle5 as pickle
import numpy as np

from PyQt5 import QtGui, QtWidgets, QtCore

from src.ui.layouts import ClassificationLayout


class EndUserWidget(QtWidgets.QWidget):
    def __init__(self):
        super(EndUserWidget, self).__init__()

        self.setObjectName("widgetEndUser")

        self.displayWidth = 800
        self.displayHeight = 600

        self.labelPreProcImage = QtWidgets.QLabel(self)
        self.labelPreProcImage.setObjectName("labelPreProcImage")
        self.labelPreProcImage.resize(self.displayWidth, self.displayHeight)
        self.labelPreProcImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.labelPreProcImage)

        self.lc = ClassificationLayout()
        self.layout.addLayout(self.lc)


class DevWidget(QtWidgets.QWidget):
    def __init__(self):
        super(DevWidget, self).__init__()

        self.setObjectName("widgetDev")

        self.displayWidth = 800
        self.displayHeight = 600

        self.labelPreProcImage = QtWidgets.QLabel(self)
        self.labelPreProcImage.setObjectName("labelPreProcImage")
        self.labelPreProcImage.setText("")
        self.labelPreProcImage.resize(self.displayWidth, self.displayHeight)
        self.labelPreProcImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.labelImage = QtWidgets.QLabel(self)
        self.labelImage.setObjectName("labelImage")
        self.labelImage.setText("")
        self.labelImage.resize(self.displayWidth, self.displayHeight)
        self.labelImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layoutVid = QtWidgets.QHBoxLayout()
        self.layoutVid.addWidget(self.labelPreProcImage)
        self.layoutVid.addWidget(self.labelImage)
        self.layout.addLayout(self.layoutVid)

        self.lc = ClassificationLayout()

        self.layout.addLayout(self.lc)


class TrainingWidget(QtWidgets.QWidget):
    def __init__(self):
        super(TrainingWidget, self).__init__()
        self.streamLength = 24
        self.streamIdx = 0
        self.recording = np.ndarray((self.streamLength, 480, 640, 3), dtype=np.uint8)

        self.setObjectName("widgetTraining")

        self.displayWidth = 800
        self.displayHeight = 600

        self.labelPreProcImage = QtWidgets.QLabel(self)
        self.labelPreProcImage.setObjectName("labelPreProcImage")
        self.labelPreProcImage.setText("")
        self.labelPreProcImage.resize(self.displayWidth, self.displayHeight)
        self.labelPreProcImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.labelCapturedImage = QtWidgets.QLabel(self)
        self.labelCapturedImage.setObjectName("labelCapturedImage")
        self.labelCapturedImage.setText("")
        self.labelCapturedImage.resize(self.displayWidth, self.displayHeight)
        self.labelCapturedImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.layout = QtWidgets.QVBoxLayout(self)

        self.layoutVid = QtWidgets.QHBoxLayout()
        self.layoutVid.addWidget(self.labelPreProcImage)
        self.layoutVid.addWidget(self.labelCapturedImage)
        self.layout.addLayout(self.layoutVid)

        self.layoutButtons = QtWidgets.QHBoxLayout()

        self.btnRec = QtWidgets.QPushButton(self)
        self.btnRec.setText("Record")
        self.btnRec.setShortcut("Space")
        self.btnRec.setCheckable(True)

        self.btnSave = QtWidgets.QPushButton(self)
        self.btnSave.setText("Save")
        self.btnSave.clicked.connect(self.saveBtn)

        self.layoutButtons.addWidget(self.btnRec)
        self.layoutButtons.addWidget(self.btnSave)

        self.layout.addLayout(self.layoutVid)
        self.layout.addLayout(self.layoutButtons)

    def recBtn(self):
        if self.btnRec.isChecked():
            print(">>> Start Recording")
            self.streamIdx = 0
            self.recording = np.zeros((self.streamLength, 480, 640, 3), dtype=np.uint8)
        else:
            print(">>> Button Not Clicked")

    def saveBtn(self):
        print(">>> Save")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "exports", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            with open(fileName, "bx") as fd:
                pickle.dump(np.array(self.recording), fd)
        self.recording = np.zeros((self.streamLength, 480, 640, 3), dtype=np.uint8)
