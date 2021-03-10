import numpy as np
import pickle5 as pickle
from collections import defaultdict
from glob import iglob


class databaseAccess:
    def __init__(self, MAIN_PATH="./database/"):
        self.path = MAIN_PATH

    def queryDB(self):
        # get the current information regarding the database
        # we need what gestures exist, and how many datapoints exist per-gest

        # get the path to the raw dataset
        # then get an iterable object of all the names using iglob
        # build an array of all the gestures that exist in our database
        for itr in iglob(self.path+"raw/*"):
            # get an iterable object for the sub-dir
            # then find the number of elements in that genorator
            self.gest[itr.split('/')[-1]] = len([i for i in iglob(itr+"/*")])

        return "foo"

    def get(self, indx, gest):
        with open("./database_legacy/raw_legacy/"+gest+"/"+str(indx),
                  "br") as fd:
            data = pickle.load(fd)

        return data

    def loadLegacy(self, dataset=-1):
        path = "./database_legacy/datasets/"+str(dataset)+"/dataset"
        
        print(path)
        with open(path, "rb") as fd:
            ((x, y), (xv, yv)) = pickle.load(fd)

        return x, y, xv, yv
