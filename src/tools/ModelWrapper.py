import tensorflow as tf
from tensorflow import keras
import numpy as np
from .preproc import PreProc
import cv2

class Predict:
    def __init__(self):
        self.conf = 0
    
    def predict_best3(self, model_in, input_to_model):
        predictions = model_in.predict(input_to_model)
        
        out = []
        # can't wrap from negative to positive slice, ae -3:0
        # and because the latter end is non-inclusive (0 should return -1), this just doesn't work
        index_sort = predictions.argsort() 
        best_three = [index_sort[0][-1], index_sort[0][-2], index_sort[0][-3]]


        for indx in best_three:
            self.conf = predictions[0][indx]*100
            char_out = chr(ord('@')+indx)
            if char_out == '@':
                char_out = 'No Obj Det (ZS)'
            out.append(str("Prediction Character: " + char_out) + "  Confidence Value: " + str(round(self.conf, 2)))

        return out


class CapModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/dog')
        self.sequence_len = 14
        self.image_sequence = np.zeros((1, self.sequence_len, 48, 64, 1))
        self.pp = PreProc()
        self.pred = Predict()

    def Classify(self, image):

        # reshape the image so that it has a channel
        image_w_channel = np.reshape(image, (np.shape(image)[0], np.shape(image)[1], 1))
        self.image_sequence = np.roll(self.image_sequence, shift=-1, axis=1)
        self.image_sequence[0][self.sequence_len-1] = image_w_channel
        
        return self.pred.predict_best3(self.model, self.image_sequence)

class CapModelTks:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/tks')
        self.sequence_len = 3
        self.image_sequence = np.zeros((1, self.sequence_len, 48, 64, 1))
        self.pp = PreProc()
        self.pred = Predict()
        self.arr = np.array([np.ones((48, 64)), np.ones((48, 64)), np.ones((48, 64))])

    def Classify(self, image):


        # reshape the image so that it has a channel
        #image_w_channel = np.reshape(image, (np.shape(image)[0], np.shape(image)[1], 1))
	
	# shift the array
        self.arr[0], self.arr[1], self.arr[2] = self.arr[1], self.arr[2], image


	# compress into one image
        input_for_tks = np.array([self.arr[0], self.arr[1], self.arr[2]], dtype=float)
        input_for_tks = np.reshape(input_for_tks, (1, 48, 64, 3))

        return self.pred.predict_best3(self.model, input_for_tks)

# this is a debug method, used for tseting the model wrapper without any other files
if ( __name__ == "__main__" ):
    cont = CapModel()
    db = DBM.databaseAccess()
    x_train, y_train, x_test, y_test = db.loadLegacy(dataset=5)

    x_train = [[[[1-x_train[i][j][k][l] for l in range(1)] for k in range(64)] for j in range(48)] for i in range(len(x_train))]
    x_test = [[[[1-x_test[i][j][k][l] for l in range(1)] for k in range(64)] for j in range(48)] for i in range(len(x_test))]

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    #test = np.expand_dims(x_test[0], axis=0)
    #print(cont.model.predict(test))
    print(cont.Classify(x_test[0]))
