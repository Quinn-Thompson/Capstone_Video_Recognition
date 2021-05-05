import time

from PyQt5 import QtCore
from cv2 import cv2
import numpy as np


class VideoThread(QtCore.QThread):
    signalUpdateImage = QtCore.pyqtSignal(np.ndarray)
    speak = QtCore.pyqtSignal(str)

    def __init__(self):
        super(VideoThread, self).__init__()
        self.imExp = 100
        self.imLife = time.time() * 1000 - self.imExp

    def run(self):
        # TODO: Get feed from intel realsense
        while True:
            if self.imExp + self.imLife <= time.time() * 1000:
                cap = cv2.VideoCapture(0)
                ret, cvImg = cap.read()
                if ret:
                    self.signalUpdateImage.emit(cvImg)
