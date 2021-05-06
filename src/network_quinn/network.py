import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle5 as pickle
import cv2
import random
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle
import math


from tensorflow.keras import layers, losses, optimizers, Input
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence as seq
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical, plot_model
import DBM


_DATA_FORMAT = 'channels_first' 


class DataGenerator(seq):
    '''
        this class is handled using a lot of function overloading, I believe
        determination of how the generation will parse batches
        essentially, pass the contents of a folder (specifically a list of file names)
        it will then 
        Heavily influenced by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        by Afshine Amidi and Shervine Amidi
    '''
    def __init__(self, path, file_names, num_classes, labels= None, dimensions=(224,224), batch_size=16,
                n_channels=1, shuffle=True, use_file_labels=False, use_file_figures=False):
        '''
            just the initialization of each of the variables used in the class
        '''
        # dimensions of the data (1d? 2d? 3? size?)
        self.dimensions = dimensions 
        # size of each of the batches that the NN will parse
        self.batch_size = batch_size 
        self.use_file_labels = use_file_labels
        # used for figures
        self.labels = labels 
        self.use_file_figures = use_file_figures
        # used for figures
        self.file_names = file_names
        # number of channels the NN will use (color? 3d?)
        self.n_channels = n_channels
        # shuffle data
        self.shuffle = shuffle 
        self.path = path
        self.num_classes = num_classes
        
        self.on_epoch_end()
    
    def __len__(self):
        '''
            parsed when the NN queries for the length of the actual batch
        '''
        # number of batches per epoch
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        '''
            parses the index of the epoch to return a set of data back to the NN
        '''
        # sfni = shuffled file names indeces
        # slices the total shuffled file list into a set of indeces that correspond to the current batch
        # this approach is taken for iterator shuffling as it does not manipulate the base data
        sfni_iterated_slice = self.shuffled_file_names_indeces[index*self.batch_size:(index+1)*self.batch_size]
        # this obtains the file names based on the passed batch of indexes
        shuffled_file_names_batch  = [self.file_names[x] for x in sfni_iterated_slice]
        # obtain the figures and the labels
        figures, labels = self.acquire_data(shuffled_file_names_batch)

        # why not set this up earlier so not to use a where in generator load? (slowish)
        # cause I am going to test a custom loss function, but still need to make a 
        # baseline model for the beta



        return figures, labels

    def on_epoch_end(self):
        '''
            at the end of each epoch and after the definition
        '''
        # arrange a list of indeces (0, 1, 2 ..., n) of the length of the number of files
        self.shuffled_file_names_indeces = np.arange(len(self.file_names))
        if self.shuffle == True:
            # randomly shuffle the indeces
            np.random.shuffle(self.shuffled_file_names_indeces)
    
    def acquire_data(self, shuffled_file_names_batch):
        # loaded array as placeholder for batching the CNN
        if _DATA_FORMAT == 'channels_first':
          loaded_figures = np.empty((self.batch_size, self.n_channels,  *self.dimensions))  
        else:
          loaded_figures = np.empty((self.batch_size, *self.dimensions, self.n_channels))  
        loaded_labels = np.empty((self.batch_size, self.num_classes), dtype=int)

        # for each file in the file batch
        for i, file_name in enumerate(shuffled_file_names_batch):
            # load the file (I believe this can be done with pickles, instead using allow_pickle=true in np.load)
            with open(self.path + file_name, "rb") as fd:
              loaded_figure_no_channel = pickle.load(fd)
            
            # reshape it so that it includes the number of channels (this may need to be reworked)
            # I would instead save the files with channels attached, instead of loading a file into each channel
            loaded_figures[i,] = loaded_figure_no_channel
            loaded_labels[i,] = to_categorical(self.labels[int(file_name)], 27)

        return loaded_figures, loaded_labels



#####
# Layer definitions
#####

# why is this distinct? residual addition should happen before activation and normalization
def conv_layer_res_end(prev_layer,
                       residual,
                       filter_size=16, 
                       kernel_size=(3,3),
                       pad_type='same',
                       activation='relu',
                       dropout=0.0):

  x = layers.Conv2D(filter_size, kernel_size, padding=pad_type, data_format=_DATA_FORMAT)(prev_layer) # densely connects x neurons with the flat image

  x = layers.Add()([x, residual])
  # normalize values between layers so that extremes do not cause rampant exponential increase in weights.
  # this is a more likely case when the activations are monotonic like relu
  # The normalization limits weight increasing and exact weight fitting
  x = layers.BatchNormalization()(x)
  
  # dropout applied before activation as to force dropped neurons into the less computationally expensive 
  # half when activation occurs (x=0 as opposed to x=x)
  if dropout != 0.0:
    if activation == 'relu':
      x = layers.Dropout(dropout)(x)

  output_conv2d = layers.Activation(activation)(x)
  
  if dropout != 0.0:
    if activation != 'relu':
      output_conv2d = layers.Dropout(dropout)(output_conv2d)  

  return output_conv2d

#convolutional layer
def conv_layer(prev_layer,
              filter_size=16, 
              kernel_size=(3,3), 
              pad_type='same',
              activation='relu',
              dropout = 0.0,
              strides = (1,1)):
  
  # simple conv2d layer
  x = layers.Conv2D(filter_size, kernel_size, padding=pad_type, strides=strides, data_format=_DATA_FORMAT)(prev_layer) 
  # normalize values between layers so that extremes do not cause rampant exponential increase in weights.
  # this is a more likely case when the activations are monotonic like relu
  # The normalization limits weight increasing and exact weight fitting
  x = layers.BatchNormalization()(x)

  # dropout applied before activation as to force dropped neurons into the less computationally expensive 
  # half when activation occurs (x=0 as opposed to x=x)
  if dropout != 0.0:
    if activation == 'relu':
      x = layers.Dropout(dropout)(x)

  output_conv2d = layers.Activation(activation)(x)
  
  # otherwise, activate afterwards, as either side includes an equation
  if dropout != 0.0:
    if activation != 'relu':
      output_conv2d = layers.Dropout(dropout)(output_conv2d)  

  return output_conv2d

# sequence layer for a resnet
def resnet_seq_layer(prev_layer, 
                     filter_size=16, 
                     kernel_size=(3,3), 
                     dropout=0.0,
                     stacked=False,
                     activation='relu',
                     mid_pool=False):
  # convolutional layer, including batch norm and activation
  x = conv_layer(prev_layer=prev_layer, filter_size=filter_size, kernel_size=kernel_size)

  # if there is another resnet sequence stacked on to this one, or just a previous conv layer with the same filter size
  # we can keep the residual daisy chain going
  if stacked == True:
    residual_x = conv_layer_res_end(prev_layer=x, residual=prev_layer, filter_size=filter_size, kernel_size=kernel_size, dropout=dropout)
  # stay with the normal conv layer
  else:
    residual_x = conv_layer(prev_layer=x, filter_size=filter_size, kernel_size=kernel_size, dropout=dropout)

  # pool between the resnet sequence
  if mid_pool == True:
    residual_x = layers.MaxPool2D((2,2), data_format=_DATA_FORMAT)(residual_x)

  x = conv_layer(prev_layer=residual_x, filter_size=filter_size, kernel_size=kernel_size)
  # output conv layer is distinct as it includes an addition before batch normalization, a common aspect of some resnets
  output_resnet_seq = conv_layer_res_end(prev_layer=x, residual=residual_x, filter_size=filter_size, kernel_size=kernel_size)

  return output_resnet_seq


if __name__ == '__main__':

  # length of the time frame
  sequence_len = 14

  if not os.path.isdir('database/my_model/dog'):

    # from tensorflow.keras import layers
    if tf.test.gpu_device_name():
      print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
      gpus = tf.config.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    else:
      print("Please install GPU version of TF")

    sequence_figure_path = 'database/sequenced_figures_post_mutation/'
    sequence_label_path = 'database/sequenced_labels_post_mutation/'

    # load in the figures for the neural network
    with open(sequence_label_path + 'sequence_labels', "rb") as fd:
      labels = np.array(pickle.load(fd), dtype=np.float64)

    # get the distinct class weights  
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = dict(zip(unique, counts))
    
    for i in class_weights:
      class_weights[i] = math.log(((1/class_weights[i])*len(labels)))
    
    ####
    # Our Network
    ####
    
    resnet_input = Input(shape=(14,48,64), name='img')
    main_path = conv_layer(prev_layer=resnet_input, filter_size=32, kernel_size=(7,7))
    
    main_path = layers.MaxPool2D(2,2, data_format=_DATA_FORMAT)(main_path)
    #main_path = layers.Dropout(0.2)(main_path)
    
    main_path = resnet_seq_layer(prev_layer=main_path, filter_size=64, kernel_size=(5, 5), mid_pool=True)
    
    main_path = layers.MaxPool2D(2,2, data_format=_DATA_FORMAT)(main_path)
    #main_path = layers.Dropout(0.2)(main_path)

    main_path = resnet_seq_layer(prev_layer=main_path, filter_size=128, kernel_size=(3, 3), mid_pool=True)

    #main_path = layers.Dropout(0.2)(main_path)

    main_path = conv_layer(prev_layer=main_path, filter_size=256, kernel_size=(3,4), pad_type='valid')
    main_path = conv_layer(prev_layer=main_path, filter_size=512, kernel_size=(1,1))

    #main_path = layers.Dropout(0.2)(main_path)

    main_path = layers.Flatten(data_format=_DATA_FORMAT)(main_path) # flatten it into 1D neuron layer

    main_path = layers.Dense(512)(main_path)

    #main_path = layers.Dropout(0.5)(main_path)

    main_path = layers.Activation('relu')(main_path) 
    
    resnet_output = layers.Dense(27, activation='softmax')(main_path) # densely connected layer that is the multiple of the image
    
    ####
    # Our Model End
    ####

    file_names = os.listdir(sequence_figure_path)
    file_array_shuffle = random.sample( file_names, len(file_names) )
    training_files=file_names[:int(len(file_array_shuffle)*0.95)]
    validation_files=file_names[int(len(file_array_shuffle)*0.95):]

    training_generator = DataGenerator(path=sequence_figure_path, dimensions=(48,64), labels=labels, batch_size=32, file_names=training_files, n_channels=sequence_len, num_classes=27, shuffle=True, use_file_labels=True)
    validation_generator = DataGenerator(path=sequence_figure_path, dimensions=(48,64), labels=labels, batch_size=32, file_names=validation_files, n_channels=sequence_len, num_classes=27, shuffle=True, use_file_labels=True)

    resnet_video = Model(inputs=resnet_input, outputs=resnet_output)

    resnet_video.summary()
    optimizer = optimizers.Adam(lr=0.00007)
    resnet_video.compile(optimizer=optimizer, loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # summarize the model in text
    resnet_video.summary()

    # generate a neat plot of the model
    plot_model(resnet_video, "database/resnet_video.png", show_shapes=True)

    # save checkpoint when val loss is lowest
    checkpoint_filepath = 'database/checkpoints/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    resnet_video.fit(training_generator,
                    epochs=40,
                    validation_data=validation_generator,
                    callbacks= [model_checkpoint_callback],
                    class_weight=class_weights)

    resnet_video.save('database/my_model/dog')
  else:
    resnet_video = load_model('database/my_model/dog')

  # The camera object
  cam = RSCW.RSC()
  pp = PP.PreProc()
  image_sequence = np.zeros((1, sequence_len, 48, 64, 1))

  while(1):
      # capture the image
      image = cam.capture()

      # proccess the image
      image = np.array(pp.preproccess(image))
      loaded_file_channel = np.reshape(image, (np.shape(image)[0], np.shape(image)[1], 1))
      image_sequence = np.roll(image_sequence, shift=-1, axis=1)
      image_sequence[0][sequence_len-1] = loaded_file_channel

      # display the image
      cv2.imshow("Depth Veiw", image)
      

      predictions = resnet_video.predict(image_sequence)
      #cv2.imshow("Depth Veiw", all_images_vali_view[i+sequence_len])
      local_and_percent = max( (v, i) for i, v in enumerate(predictions[0]) )
      print('Percent max: ' + str(int(local_and_percent[0]*100)).zfill(3) + ' Max Locale: ' + str(chr(local_and_percent[1] + 64)), end='\r')

      # if a key is pressed, start the collection, otherwise loop
      k = cv2.waitKey(100)

      # check to see if we want to leave
      # ESC == 27 in ascii
      if k == 27:
          break
