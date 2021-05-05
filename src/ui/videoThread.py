import time

from PyQt5 import QtCore
from cv2 import cv2
from ..tools.RSC_Wrapper import RSC

import numpy as np

class VideoThread(QtCore.QThread):
    signalUpdateImage = QtCore.pyqtSignal(np.ndarray)
    speak = QtCore.pyqtSignal(str)

    def __init__(self):
        super(VideoThread, self).__init__()
        self.camera = RSC()
        self.prev_time = time.time() * 1000
        # time before image expiration
        self.imExp = 100
        self.imLife = time.time() * 1000 - self.imExp

    def run(self):
        # TODO: Get feed from intel realsense
        # run forever
        while True:

            # get the captured depth image
            im_depth = self.camera.capture()
            # update image display
            # if the time taken between last check and this check is 100 milliseconds 
            if self.imExp + self.imLife <= time.time() * 1000:
                self.signalUpdateImage.emit(im_depth)
                # reset the image life
                self.imLife = time.time() * 1000
                #print("timestamp videoThread: " + str(time.time() * 1000) + ". t between last frame " + str (time.time() * 1000 - self.prev_time))
                self.prev_time = time.time() * 1000
