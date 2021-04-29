from DBM import databaseAccess
from RSC_Wrapper import RSC
from image_man import PreProc
import numpy as np
import time

class Middelware:
    def __init__(self, imageExpire=100):
        # define the helper objects
        self.dbm = databaseAccess()
        self.camera = RSC()
        self.pp = PreProc(resize=False)

        # image expire defines how old of a picture to save before taking
        # a new one
        # pased in ms stored in ns
        self.imExp = imageExpire
        self.imageLife = time.time()*1000 - self.imExp

        self.imDepth = None
        #self.ppDepth = None
        self.imRGB = None

        # at this point, we'd probably want to do something lol
        self.update()

    def update(self):
        # aquire the latest camera info if nessisary
        if (self.imExp + self.imageLife) <= time.time()*1000:
            # capture an image
            self.imDepth = self.camera.capture()
            #self.ppDepth = self.pp.preproccess(self.imDepth)

            # reset te image life
            self.imageLife = time.time() * 1000

    def returnDeptInt(self, size=(48, 64)):
        # convert to int
        img = self.imDepth
            
        # resize the image
        img = self.pp.resize(new_shape=size, image=img)

        return img

    def returnDeptIntPP(self, size=(48, 64)):
        # convert to int
        img = self.pp.preproccess(self.imDepth)
            
        # resize the image
        img = self.pp.resize(new_shape=size, image=img)

        return img
