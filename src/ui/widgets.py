from PyQt5 import QtWidgets, QtGui, QtCore

from cv2 import cv2
import numpy as np
from src.ui.layouts import ClassificationLayout


class EndUserWidget(QtWidgets.QWidget):
    def __init__(self, vt):
        super(EndUserWidget, self).__init__()

        vt.signalUpdateImage.connect(self.updateImage)

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

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, cvImg):
        rgbImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        self.labelImage.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    rgbImg.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).scaled(
                    self.displayWidth,
                    self.displayHeight,
                    QtCore.Qt.KeepAspectRatio,
                )
            )
        )


class DevWidget(QtWidgets.QWidget):
    def __init__(self, vt):
        super(DevWidget, self).__init__()

        vt.signalUpdateImage.connect(self.updateImage)

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

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, cvImg):
        rgbImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        preprocImg = self.preProcImg(cvImg)
        h, w, ch = rgbImg.shape
        self.labelImage.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    rgbImg.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).scaled(
                    self.displayWidth,
                    self.displayHeight,
                    QtCore.Qt.KeepAspectRatio,
                )
            )
        )
        self.labelPreProcImage.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    preprocImg.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).scaled(
                    self.displayWidth,
                    self.displayHeight,
                    QtCore.Qt.KeepAspectRatio,
                )
            )
        )

    def preProcImg(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TrainingWidget(QtWidgets.QWidget):
    def __init__(self, vt):
        super(TrainingWidget, self).__init__()

        vt.signalUpdateImage.connect(self.updateImage)

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

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, cvImg):
        preprocImg = self.preProcImg(cvImg)
        h, w, ch = preprocImg.shape
        self.labelPreProcImage.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    preprocImg.data, w, h, ch * w, QtGui.QImage.Format_RGB888
                ).scaled(
                    self.displayWidth,
                    self.displayHeight,
                    QtCore.Qt.KeepAspectRatio,
                )
            )
        )

    def preProcImg(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
