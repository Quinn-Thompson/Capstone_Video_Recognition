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
                # preprocess to remove background and scale from 1 to 0
                im_d_preproc = self.pp_noresize.preproccess(im_depth)
                # reset back to 0-255 for 8 bit greyscale image
                # this is done beacuse proprocess does a downscale from 0-1 while 
                # removing the background
                im_d_preprocm = np.asarray(np.multiply(im_d_preproc, 255), dtype=np.uint8)
                cv2.imshow("test", im_d_preproc)

                # if a key is pressed, start the collection, otherwise loop
                k = cv2.waitKey(1)
                # update image display
                self.signalUpdateImage.emit(im_d_preprocm)
                # reset the image life
                self.imLife = time.time() * 1000
