import functools
import os
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from cv2 import cv2
import numpy as np

from src.tools.ModelWrapper import CapModel
from src.ui.videoThread import VideoThread
from src.ui.widgets import DevWidget, EndUserWidget, TrainingWidget
from ..tools.preproc import PreProc


class MLGestureRecognition(QtWidgets.QWidget):
    def __init__(self):
        super(MLGestureRecognition, self).__init__()
        # initialize preprocessing process but disallow resizing in this case
        self.pp_noresize = PreProc(resize=False)
        self.setupUI()
        self.prev_time = time.time() * 1000

    def setupUI(self):
        self.initMainWindow()

        self.initMenuBar()
        self.initUI()
        self.mainWindow.setCentralWidget(self.stackedWidget)
        QtCore.QMetaObject.connectSlotsByName(self.mainWindow)
        self.initVideoThread()

    def initMainWindow(self):
        self.mainWindow = QtWidgets.QMainWindow()
        self.mainWindow.setGeometry(100, 100, 800, 600)
        self.mainWindow.setWindowTitle("ML Gesture Recognition")

    def initMenuBar(self):
        # Create Menu bar
        self.menubar = QtWidgets.QMenuBar(self.mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")

        self.mainWindow.setMenuBar(self.menubar)

        # File dropdown
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuFile.setTitle("File")
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionQuit = QtWidgets.QAction(self.mainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionQuit.setText("Quit")
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionQuit.triggered.connect(lambda: sys.exit())

        self.menuFile.addAction(self.actionQuit)

        # Models dropdown
        self.menuModels = QtWidgets.QMenu(self.menubar)
        self.menuModels.setObjectName("menuModels")
        self.menuModels.setTitle("Models")

        for i in os.listdir("./models"):
            model = QtWidgets.QAction(self.mainWindow)
            model.setObjectName(f"action{i}")
            model.setText(i)
            model.triggered.connect(functools.partial(self.loadModel, i))
            self.menuModels.addAction(model)

        self.menubar.addAction(self.menuModels.menuAction())
        self.loadModel("dog")

        # View dropdown
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuView.setTitle("View")

        for i in ["EndUser", "Dev", "Training"]:
            view = QtWidgets.QAction(self.mainWindow)
            view.setObjectName(f"action{i}")
            view.setText(i)
            view.triggered.connect(functools.partial(self.changeView, i))
            self.menuView.addAction(view)

        self.menubar.addAction(self.menuView.menuAction())

    def initVideoThread(self):
        self.thread = VideoThread()
        self.thread.start()
        self.thread.signalUpdateImage.connect(self.updateImage)

    def initUI(self):
        self.stackedWidget = QtWidgets.QStackedWidget(self.mainWindow)
        self.stackedWidget.setObjectName("stackedWidget")

        self.widgetEndUser = EndUserWidget()
        self.stackedWidget.addWidget(self.widgetEndUser)

        self.widgetDev = DevWidget()
        self.stackedWidget.addWidget(self.widgetDev)

        self.widgetTraining = TrainingWidget()
        self.stackedWidget.addWidget(self.widgetTraining)

        self.stackedWidget.setCurrentWidget(self.widgetEndUser)

    def loadModel(self, model):
        print(f">>> Load model: {model}")
        if model == "dog":
            self.model = CapModel()

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, cvImg):
        print("timestamp updateImage: " + str(time.time() * 1000) + ". t between last frame " + str (time.time() * 1000 - self.prev_time) + "\n")
        self.prev_time = time.time() * 1000
        cur = self.stackedWidget.currentWidget()

        preProcImgD, preProcImgN = self.preProc(cvImg)
        h, w = preProcImgD.shape
        cur.labelPreProcImage.setPixmap(
            QtGui.QPixmap.fromImage(
                QtGui.QImage(
                    preProcImgD.data, w, h, w, QtGui.QImage.Format_Grayscale8
                ).scaled(
                    cur.displayWidth,
                    cur.displayHeight,
                    QtCore.Qt.KeepAspectRatio,
                )
            )
        )

        if cur == self.widgetTraining:
            if cur.btnRec.isChecked():
                if cur.streamIdx < cur.streamLength:
                    cur.recording[cur.streamIdx] = preProcImgD
                else:
                    cur.btnRec.setChecked(False)

            if cur.streamIdx >= cur.streamLength:
                cur.streamIdx = 0

            cur.labelCapturedImage.setPixmap(
                QtGui.QPixmap.fromImage(
                    QtGui.QImage(
                        cur.recording[cur.streamIdx].data,
                        w,
                        h,
                        w,
                        QtGui.QImage.Format_Grayscale8,
                    ).scaled(
                        cur.displayWidth,
                        cur.displayHeight,
                        QtCore.Qt.KeepAspectRatio,
                    )
                )
            )
            cur.lc.updatePredictions(self.model.Classify(preProcImgN))
            cur.streamIdx += 1

        if cur == self.widgetDev or cur == self.widgetTraining:
            cvImg = np.asarray((cvImg-np.min(cvImg))/(np.max(cvImg)/255), dtype=np.uint8)
            h, w = cvImg.shape
        else:
            cur.lc.updatePredictions(self.model.getTop3(cvImg))

        if cur == self.widgetDev:
            rgbImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
            h, w = rgbImg.shape
            cur.labelImage.setPixmap(
                QtGui.QPixmap.fromImage(
                    QtGui.QImage(
                        cvImg.data, w, h, w, QtGui.QImage.Format_Grayscale8
                    ).scaled(
                        cur.displayWidth,
                        cur.displayHeight,
                        QtCore.Qt.KeepAspectRatio,
                    )
                )
            )
        print("Finished signal\n")

    # TODO: Update preProc
    def preProc(self, img):
        # this image is for displaying
        # preprocess to remove background and scale from 1 to 0
        im_preproc = self.pp_noresize.preproccess(img)
        # reset back to 0-255 for 8 bit greyscale image
        # this is done beacuse proprocess does a downscale from 0-1 while
        # removing the background
        im_d_preproc = np.asarray(np.multiply(im_preproc, 255), dtype=np.uint8)

        # this image is for the network (48 by 64 image resize)
        im_n_preproc = self.pp_noresize.resize(new_shape=(48, 64), image=im_preproc)

        return im_d_preproc, im_n_preproc

    def changeView(self, view):
        print(f">>> Change View: {view}")
        if view == "EndUser":
            self.stackedWidget.setCurrentWidget(self.widgetEndUser)
        elif view == "Dev":
            self.stackedWidget.setCurrentWidget(self.widgetDev)
        elif view == "Training":
            self.stackedWidget.setCurrentWidget(self.widgetTraining)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MLGestureRecognition()
    ui.mainWindow.show()
    sys.exit(app.exec_())
