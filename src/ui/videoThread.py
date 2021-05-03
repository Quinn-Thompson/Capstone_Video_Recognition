from PyQt5 import QtCore

import numpy as np
import time
from cv2 import cv2


class VideoThread(QtCore.QThread):
    signalUpdateImage = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(VideoThread, self).__init__()
        self.imExp = 100
        self.imLife = time.time() * 1000 - self.imExp

    def run(self):
        while True:
            if self.imExp + self.imLife <= time.time() * 1000:
                cap = cv2.VideoCapture(-1)
                ret, cvImg = cap.read()
                if ret:
                    self.signalUpdateImage.emit(cvImg)
