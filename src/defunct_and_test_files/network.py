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

import keras.callbacks as cbks

from tensorflow.keras import layers, losses, optimizers, Input, metrics, callbacks
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence as seq
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical, plot_model
from tkinter import Tk, Label, Button, StringVar, LEFT, RIGHT, BOTTOM, TOP, NW, SW, NE, SE, W, N

import DBM

import keras.backend as K
from PIL import Image, ImageTk

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
        loaded_labels = np.empty((self.batch_size, self.num_classes + 2), dtype=np.float32)

        # for each file in the file batch
        for i, file_name in enumerate(shuffled_file_names_batch):
            # load the file (I believe this can be done with pickles, instead using allow_pickle=true in np.load)
            with open(self.path + file_name, "rb") as fd:
              loaded_figure_no_channel = pickle.load(fd)
            
            # reshape it so that it includes the number of channels (this may need to be reworked)
            # I would instead save the files with channels attached, instead of loading a file into each channel
            loaded_figures[i,] = loaded_figure_no_channel
            # then insert a label based on the file name without the .npy ( as mine were .npy array file, fastest loading time but larger than pickle)
            loaded_labels_ns = self.labels[int(file_name)]
            loaded_labels[i,2:]=to_categorical(loaded_labels_ns[2], 26)
            loaded_labels[i,0:2]=loaded_labels_ns[0:2]

        return loaded_figures, loaded_labels


def test_loss(label_actual, label_pred):
  
  dif = label_pred[:,2:26] - label_actual[:,2:26]
  square = np.square(dif)
  sum_square = _LAMBDA_STILL*np.sum(square,axis=1)
  exist = label_actual[:,0]*sum_square
  sum_exist = np.sum(exist)

  dif = label_pred[:,0] - label_actual[:,0]
  square = np.square(dif)
  exist = label_actual[:,0]* square
  sum_exist = _LAMBDA_PSTILL*np.sum(exist)

  dif = label_pred[:,0] - label_actual[:,0]
  square = np.square(dif)
  exist = (1-label_actual[:,0])* square
  sum_exist = _LAMBDA_PSTILL*np.sum(exist)


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

#defined outside of loss to prevent reassignment every batch

# existence of 
_LAMBDA_STILL = 1.2
# 2 movement to every 24 still, meaining probability is ~2/26, so to compensate w multiply by 13
# punishing multiplier for probability neuron not existing
_LAMBDA_PNOSTIL = 0.8
# punishing multiplier for probability neuron existing
_LAMBDA_PSTILL = 1.2
_LAMBDA_MOVE = 1.5
_LAMBDA_NOMOVE = 1.3

# custom you only look once v1 
# don't know why everyone uses Y and X for machine learning, makes everything hella confusing
# as opposed to labels and figures, which are unique terms
def loss(class_weights):
  def custom_loss(label_actual, label_pred):
    
    # penalize class based on existence and square difference 
    penalize_still = K.sum(_LAMBDA_STILL*(label_actual[:,0]*K.sum(K.square(label_pred[:,2:26] - label_actual[:,2:26]), axis=1)))

    penalize_move = K.sum(label_actual[:,1]*(K.sum(K.square(label_pred[:,26:28] - label_actual[:,26:28] ),axis=1)))

    # adjust probablitiy if object exists, only when the 0th neuron is 1
    penalize_probability_still = _LAMBDA_PSTILL*K.sum(label_actual[:,0]*(K.square(label_pred[:,0] - label_actual[:,0])))
    # penalize when the probability neuron is not close to 0, when a known non-object is in the grid
    # makes sure to adjust the probability if no object exists at 0.5 rate, as there are many more instances where this is true
    penalize_probability_nostill = _LAMBDA_PNOSTIL*K.sum((1-label_actual[:,0])*(K.square(label_pred[:,0] - label_actual[:,0])))

    penalize_probability_move = _LAMBDA_MOVE*K.sum(K.square(label_pred[:,1] - label_actual[:,1]))  
    # penalize when the probability neuron is not close to 0, when a known non-object is in the grid
    # makes sure to adjust the probability if no object exists at 0.5 rate, as there are many more instances where this is true
    penalize_probability_nomove = _LAMBDA_NOMOVE*K.sum(K.sqrt(1-label_actual[:,1])*(K.square(label_pred[:,1] - label_actual[:,1])))

    # sum together for a single loss value, allowing for back propogation
    return (penalize_still+ penalize_move + penalize_probability_still + penalize_probability_nostill + 
    penalize_probability_move + penalize_probability_nomove)
    
    
    
    # + penalize_move  penalize_probability_move + 
                                    #penalize_probability_nomove)
  return custom_loss

class return_label_act(metrics.Metric):
    def __init__(self, name="return_label_act", **kwargs):
        super(return_label_act, self).__init__(name=name, **kwargs)
        self.label_actual = None

    def update_state(self, label_actual, label_pred, sample_weight=None):
        self.label_actual = label_actual

    def result(self):
        return self.label_actual

# why is there a dedicated class for this?
# tensroflow backend automatically takes the mean of simple metrics
# this means the outpud tensor will be a single scalar
class return_label_pred(metrics.Metric):
    def __init__(self, name="return_label_pred", **kwargs):
        super(return_label_pred, self).__init__(name=name, **kwargs)
        self.label_pred = None

    def update_state(self, label_actual, label_pred, sample_weight=None):
        self.label_pred = label_pred

    def result(self):
        return self.label_pred

def ACC(label_actual, label_pred):
  return K.sum(K.cast((label_actual[:,0] * K.cast(K.argmax(label_actual[:,2:28], axis=1), 'float32')) == (label_actual[:,0] * K.cast(K.argmax(label_pred[:,2:28], axis=1), 'float32')), 'float32')) / 32

# Mean Squared Error of the probability of known non-object grids 
def MSE_GESACCR_STL(label_actual, label_pred):
  # very explicit because otherwise it's hard to read
  difference_in_real = label_pred[:,2:26] - label_actual[:,2:26]

  square_difference = K.square(difference_in_real)

  true_gesture = label_actual[:,2:26] * square_difference

  sum_true_gestures = K.sum(true_gesture, axis=1)

  mean_true_gesutres = sum_true_gestures / (K.sum(label_actual[:,2:26]) + 1)

  true_objects = label_actual[:,0] * mean_true_gesutres 

  sum_true_objects = K.sum(true_objects)

  mean_true_object = sum_true_objects / (K.sum(label_actual[:,0])+1)

  return mean_true_object

def MSE_NONGESACCR_STL(label_actual, label_pred):
    # very explicit because otherwise it's hard to read
  difference_in_real = label_pred[:,2:26] - label_actual[:,2:26]

  square_difference = K.square(difference_in_real)

  false_gesture = (1-label_actual[:,2:26]) * square_difference

  sum_false_gestures = K.sum(false_gesture, axis=1)

  mean_false_gesutres = sum_false_gestures / (K.sum(1-label_actual[:,2:26]) + 1)

  true_objects = label_actual[:,0] * mean_false_gesutres 

  sum_true_objects = K.sum(true_objects)

  mean_true_object = sum_true_objects / (K.sum(label_actual[:,0])+1)

  return mean_true_object

def MSE_ACCR_MOV(label_actual, label_pred):
  return K.sum(label_actual[:,1] * (K.sum(K.square(label_pred[:,26:28] - label_actual[:,26:28]), axis=1))) / (K.sum(label_actual[:,1])+1)

def MSE_KN_STL(label_actual, label_pred):
  return K.sum(label_actual[:,0] * (K.square(label_pred[:,0] - label_actual[:,0]))) / (K.sum(label_actual[:,0])+1)

def MSE_NKN_STL(label_actual, label_pred):
  return K.sum((1-label_actual[:,0]) * (K.square(label_pred[:,0] - label_actual[:,0]))) / (K.sum(1-label_actual[:,0])+1)

def MSE_KN_MOV(label_actual, label_pred):
  return K.sum(K.square(label_pred[:,1] - label_actual[:,1])) / (K.sum(label_actual[:,1])+1)

def MSE_NKN_MOVE(label_actual, label_pred):
  return K.sum((1-label_actual[:,1]) * (K.square(label_pred[:,1] - label_actual[:,1]))) / (K.sum(1-label_actual[:,1])+1)


# global just so I don't have to keep scrolling up and down
global_single_items = ['batch', 'epoch', 'loss', 'val_loss']
global_single_image_items = ['label_dif']
global_simple_custom_metrics = [ACC, MSE_GESACCR_STL, MSE_NONGESACCR_STL, MSE_ACCR_MOV, MSE_KN_STL, MSE_NKN_STL, MSE_KN_MOV, MSE_NKN_MOVE]
global_object_custom_metrics = [return_label_pred(), return_label_act()]

def scheduler(epoch, lr):
  if epoch < 6:
    return lr
  if epoch >= 6:
    return (lr * 0.5)

class load_to_tkinter(cbks.Callback):

  def __init__(self, label_dict_single, label_dict_single_image, label_dict_simple, label_dict_object, **kwargs):
    super(load_to_tkinter, self).__init__()
    self._supports_tf_logs = True
    self.label_dict_single = label_dict_single
    self.label_dict_simple = label_dict_simple
    self.label_dict_object = label_dict_object
    self.label_dict_single_image = label_dict_single_image
    # keeps images in existence
    self.label_image_tk={}
    self.numpy_label={}


  def convert_image(self, string_var, numpy_label):
      label_image = Image.fromarray(numpy_label)
      self.label_image_tk[string_var] = ImageTk.PhotoImage(image=label_image) 
      root.label_image_tk = self.label_image_tk[string_var]


  # tkinter will freeze because of eopoch length, needs idle task update after each batch
  # can do this call if I can figure out how to pass root through
  def on_batch_end(self, batch, logs=None):
    if batch % 50 == 0:
      for string_var in self.label_dict_simple.keys():
        if not string_var.startswith('val_'):
          self.label_dict_simple[string_var].configure(text=string_var + ': ' + str(logs[string_var].numpy()))

      for string_var in self.label_dict_object.keys():
        if not string_var.startswith('val_'):
          self.numpy_label[string_var] = cv2.resize(logs[string_var].numpy()*256, (32*8,28*8))
          self.convert_image(string_var, self.numpy_label[string_var])
          self.label_dict_object[string_var].configure(image=self.label_image_tk[string_var])

      self.convert_image('label_dif', np.abs(np.subtract(self.numpy_label['return_label_pred'], self.numpy_label['return_label_act'])))
      self.label_dict_single_image['label_dif'].configure(image=self.label_image_tk['label_dif'])

    self.label_dict_single['loss'].configure(text='LOSS' + ': ' + str(logs['loss'].numpy()))
    self.label_dict_single['batch'].configure(text='batch#' + ': ' + str(batch))
    #root.after(1000)


    root.update()
    root.update_idletasks()

  def on_epoch_end(self, epoch, logs=None):
    for string_var in self.label_dict_simple.keys():
      if string_var.startswith('val_'):
        self.label_dict_simple[string_var].configure(text=string_var + ': ' + str(logs[string_var]))

    self.label_dict_single['val_loss'].configure(text='VAL_LOSS' + ': ' + str(logs['val_loss']))
    self.label_dict_single['epoch'].configure(text='EPOCH#' + ': ' + str(epoch+1))

    root.update()
    root.update_idletasks()
    

class train_gui:

  def __init__(self, master):
      self.master = master
      master.title("A simple GUI")

      single_item = global_single_items
      single_item_image = global_single_image_items
      custom_metrics = global_simple_custom_metrics
      object_metric = global_object_custom_metrics

      self.passed_metrics = custom_metrics + object_metric

      self.label_dict_single = {}
      self.label_dict_simple_image = {}
      self.label_dict_simple = {}
      self.label_dict_object = {}

      for each_metric in single_item:
        self.label_dict_single[each_metric] = Label(master, text='0')
        self.label_dict_single[each_metric].pack(side=TOP, anchor=NW)

      for each_metric in custom_metrics:
        self.label_dict_simple[each_metric.__name__] = Label(master, text='0')
        self.label_dict_simple[each_metric.__name__].pack(side=TOP, anchor=NW)

      for each_metric in custom_metrics:
        self.label_dict_simple['val_' + each_metric.__name__] = Label(master, text='0')
        self.label_dict_simple['val_' + each_metric.__name__].pack(side=TOP, anchor=NW)

      zero_np = np.zeros((32*8,28*8))
      zero_image = Image.fromarray(zero_np)
      zero_image_tk = ImageTk.PhotoImage(image=zero_image) 
      root.zero_image_tk = zero_image_tk

      for each_metric in single_item_image:
        self.label_dict_simple_image[each_metric] = Label(master, image=root.zero_image_tk)
        self.label_dict_simple_image[each_metric].pack(side=RIGHT, anchor=NE)

      for each_metric in object_metric:
        self.label_dict_object[type(each_metric).__name__] = Label(master, image=root.zero_image_tk)
        self.label_dict_object[type(each_metric).__name__].pack(side=RIGHT, anchor=NE)


      self.train_button = Button(master, text="Train", command=self.train_nn)
      self.train_button.pack()

      self.close_button = Button(master, text="Close", command=master.quit)
      self.close_button.pack()

  def train_nn(self):
    #self.string_var_dict['MSE_GESACCR_STL'].set('fmf')
    #root.update_idletasks()
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
      class_weights = np.zeros((2))
      #unique, counts = np.unique(labels[:, 0:2], return_counts=True)
      class_weights[0] = np.sum(labels[:, 0]/24, axis = 0)
      class_weights[1] = np.sum(labels[:, 0]/2, axis = 0)
      
      for i, weight in enumerate(class_weights):
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
      
      resnet_output = layers.Dense(28)(main_path) # densely connected layer that is the multiple of the image

      ####
      # Our Model End
      ####

      file_names = os.listdir(sequence_figure_path)
      file_array_shuffle = random.sample( file_names, len(file_names) )
      training_files=file_names[:int(len(file_array_shuffle)*0.95)]
      validation_files=file_names[int(len(file_array_shuffle)*0.95):]

      training_generator = DataGenerator(path=sequence_figure_path, dimensions=(48,64), labels=labels, batch_size=32, file_names=training_files, n_channels=sequence_len, num_classes=26, shuffle=True, use_file_labels=True)
      validation_generator = DataGenerator(path=sequence_figure_path, dimensions=(48,64), labels=labels, batch_size=32, file_names=validation_files, n_channels=sequence_len, num_classes=26, shuffle=True, use_file_labels=True)

      resnet_video = Model(inputs=resnet_input, outputs=resnet_output)

      resnet_video.summary()
      optimizer = optimizers.Adam(lr=0.0001)
      resnet_video.compile(optimizer=optimizer, 
                           loss=loss(np.array(class_weights)), 
                           metrics=self.passed_metrics)

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
      
      lr_schedular = callbacks.LearningRateScheduler(scheduler)

      resnet_video.fit(training_generator,
                      epochs=15,
                      validation_data=validation_generator,
                      callbacks= [model_checkpoint_callback,
                                  lr_schedular,
                                  load_to_tkinter(self.label_dict_single, self.label_dict_simple_image, self.label_dict_simple, self.label_dict_object)])

      resnet_video.optimizer = None
      resnet_video.compiled_loss = None
      resnet_video.compiled_metrics = None


      resnet_video.save('database/my_model/dog')
    else:
      resnet_video = load_model('database/my_model/dog')

    # The camera object
    cam = RSCW.RSC()
    pp = PP.PreProc()
    image_sequence = np.zeros((1, sequence_len, 48, 64, 1))

    prev_best_guess = ' '

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
        local_and_percent = max( (v, i) for i, v in enumerate(predictions[0][2:28]) )
        #print('Percent max: ' + str(int(local_and_percent[0]*100)).zfill(3) + ' Max Locale: ' + str(chr(local_and_percent[1] + 63))
        #+ ' Confidence Neuron: ' + str(predictions[0][0] * 100) + ' Rolling Confidence Neuron: ' + str(predictions[0][1] * 100), end='\r')

        if predictions[0][0] > 0.7 and local_and_percent[1] < 24:
          prev_best_guess = str(chr(local_and_percent[1] + 65))
          print('Letter: ' + str(chr(local_and_percent[1] + 65)) + '          ', end='\r')
        elif predictions[0][1] > 0.85 and local_and_percent[1] >= 24:
          prev_best_guess = str(chr(local_and_percent[1] + 65))
          print('Letter: ' + str(chr(local_and_percent[1] + 65) + '?          '), end='\r')
        elif predictions[0][0] > 0.7 and predictions[0][1] > 0.85 and local_and_percent[1] >= 24:
          prev_best_guess = str(chr(local_and_percent[1] + 65))
          print('Letter: ' + str(chr(local_and_percent[1] + 65)) + '          ', end='\r')
        else:
          print('Nan, Prev: ' + prev_best_guess + '            ', end='\r')


        # if a key is pressed, start the collection, otherwise loop
        k = cv2.waitKey(100)

        # check to see if we want to leave
        # ESC == 27 in ascii
        if k == 27:
            break

root = Tk()

if __name__ == '__main__':
  my_gui = train_gui(root)
  root.geometry('800x800')
  

  root.mainloop()