from PyQt5 import QtGui, QtWidgets

from src.ui.layouts import ClassificationLayout


class EndUserWidget(QtWidgets.QWidget):
    def __init__(self):
        super(EndUserWidget, self).__init__()

        self.setObjectName("widgetEndUser")

        self.displayWidth = 800
        self.displayHeight = 600

        self.labelImage = QtWidgets.QLabel(self)
        self.labelImage.setObjectName("labelImage")
        self.labelImage.resize(self.displayWidth, self.displayHeight)
        self.labelImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.labelImage)

        self.lc = ClassificationLayout()
        self.layout.addLayout(self.lc)


class DevWidget(QtWidgets.QWidget):
    def __init__(self):
        super(DevWidget, self).__init__()

        self.setObjectName("widgetDev")

        self.displayWidth = 800
        self.displayHeight = 600

        self.labelImage = QtWidgets.QLabel(self)
        self.labelImage.setObjectName("labelImage")
        self.labelImage.setText("")
        self.labelImage.resize(self.displayWidth, self.displayHeight)
        self.labelImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.labelPreProcImage = QtWidgets.QLabel(self)
        self.labelPreProcImage.setObjectName("labelPreProcImage")
        self.labelPreProcImage.setText("")
        self.labelPreProcImage.resize(self.displayWidth, self.displayHeight)
        self.labelPreProcImage.setPixmap(QtGui.QPixmap("bk.jpg"))

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layoutVid = QtWidgets.QHBoxLayout(self)
        self.layoutVid.addWidget(self.labelImage)
        self.layoutVid.addWidget(self.labelPreProcImage)
        self.layout.addLayout(self.layoutVid)

        self.lc = ClassificationLayout()

        self.layout.addLayout(self.lc)


class TrainingWidget(QtWidgets.QWidget):
    def __init__(self):
        super(TrainingWidget, self).__init__()

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
        self.layoutVid = QtWidgets.QHBoxLayout(self)
        self.layoutVid.addWidget(self.labelPreProcImage)
        self.layoutVid.addWidget(self.labelCapturedImage)
        self.layout.addLayout(self.layoutVid)
