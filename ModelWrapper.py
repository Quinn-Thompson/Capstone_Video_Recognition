import tensorflow as tf
import DBM
from tensorflow import keras
import numpy as np

class CapModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('my_model/dog')
        self.sequence_len = 14
        self.image_sequence = np.zeros((1, self.sequence_len, 48, 64, 1))


    def Classify(self, image):
        # reshape the image so that it has a channel
        image_w_channel = np.reshape(image, (np.shape(image)[0], np.shape(image)[1], 1))
        self.image_sequence = np.roll(self.image_sequence, shift=-1, axis=1)
        self.image_sequence[0][self.sequence_len-1] = image_w_channel


        predictions = self.model.predict(self.image_sequence)

        indx = np.argmax(predictions)
        return chr(ord('@')+indx), predictions[0][indx]*100

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