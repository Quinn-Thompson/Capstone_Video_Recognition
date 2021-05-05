import time

from PyQt5 import QtCore
from cv2 import cv2
from ..tools.RSC_Wrapper import RSC
from ..tools.preproc import PreProc

import numpy as np

class VideoThread(QtCore.QThread):
    signalUpdateImage = QtCore.pyqtSignal(np.ndarray)
    speak = QtCore.pyqtSignal(str)

    def __init__(self):
        super(VideoThread, self).__init__()
        self.camera = RSC()
        # initialize preprocessing process but disallow resizing in this case
        self.pp_noresize = PreProc(resize=False)
        
        # time before image expiration
        self.imExp = 100
        self.imLife = time.time() * 1000 - self.imExp

    def run(self):
        # TODO: Get feed from intel realsense
        # run forever
        while True:
            # if the time taken between last check and this check is 100 milliseconds 
            if self.imExp + self.imLife <= time.time() * 1000:
                # get the captured depth image
                im_depth = self.camera.capture()
                # update image display
                self.signalUpdateImage.emit(im_d_preprocm)
                # reset the image life
                self.imLife = time.time() * 1000
