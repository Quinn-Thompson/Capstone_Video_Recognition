import matplotlib.pyplot as plt
import numpy as np
import os
import pickle5 as pickle
import cv2
import random
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle

from shutil import rmtree as remove_folder

def roll_sequence(file_path_open, 
                  file_path_save, 
                  sequence_array, 
                  label_new, 
                  label_old, 
                  sequence_len, 
                  current_figure,
                  first):
    directory_listing = os.listdir(file_path_open)
    for i, figure in enumerate(directory_listing):
        with open(file_path_open+figure, "rb") as fd:
            sequence_array = np.roll(sequence_array, shift=-1, axis=0)
            loaded_file = pickle.load(fd)
            sequence_array[sequence_len-1] = loaded_file
        
        if i >= sequence_len or first == False:
            label_new.append(label_old[i])
            with open(file_path_save + str(current_figure).zfill(6), "bx") as fd:
                pickle.dump(sequence_array, fd)
            current_figure += 1 
    #print(sequence_array)
    return sequence_array, current_figure

def sequence_data():
    # paths to load figures and labels from

    gesture_path = 'database/gestures_seperate/'
    labels_path = 'database/labels_seperate/'


    figure_save_path = 'database/sequenced_figures_post_mutation'
    label_save_path = 'database/sequenced_labels_post_mutation'

    if os.path.isdir(figure_save_path):
        print('resetting sequenced figure folder')
        remove_folder(figure_save_path)
    os.mkdir(path = figure_save_path)

    if os.path.isdir(label_save_path):
        print('resetting sequenced label folder')
        remove_folder(label_save_path)
    os.mkdir(path = label_save_path)

    


    # length of the lstm sequence
    sequence_len = 14

    current_figure = 0

    folder_gestures = os.listdir(gesture_path)

    labels_new = []

    # yay nested
    for gestures in folder_gestures:
        print(gestures)
        with open(labels_path + gestures + '/labels', "rb") as fd:
            labels = np.array(pickle.load(fd), dtype=np.float64)

        gesture_mutations_path = gesture_path + gestures + "/gesture_mutations/"
        current_gesture_mutation_names = os.listdir(gesture_mutations_path)
        # for each gesture mutations within these gestures
        for gesture_mutations in current_gesture_mutation_names:
            # figures from one of the files
            random_gesture_partition = folder_gestures.copy()    

            for i in range(1):
                random.shuffle(random_gesture_partition)
                random_mutation_path = gesture_path + random_gesture_partition[0] + "/gesture_mutations/"
                with open(labels_path + random_gesture_partition[0] + '/labels', "rb") as fd:
                    labels_random_partition = np.array(pickle.load(fd), dtype=np.float64)
                #random_gesture_partition = random_gesture_partition[:-1]
                random_mutation_partition = os.listdir(random_mutation_path)
                random.shuffle(random_mutation_partition)
                


                # for some reason this array is acting immutable when passed to the roll sequence function?
                # it's really dumb, as it does it even when the write flag is set to true
                # i think it may be np.roll
                # that's why it's being passed back
                sequence_array = np.zeros(shape=(sequence_len, 48, 64))
                sequence_array.setflags(write=1)
                sequence_array, current_figure = roll_sequence((random_mutation_path+random_mutation_partition[0] + "/"), 
                                                figure_save_path + '/', 
                                                sequence_array, 
                                                labels_new, 
                                                labels_random_partition, 
                                                sequence_len,
                                                current_figure,
                                                True)
                _, current_figure = roll_sequence((gesture_mutations_path+gesture_mutations + "/"), 
                                                figure_save_path + '/', 
                                                sequence_array, 
                                                labels_new, 
                                                labels, 
                                                sequence_len,
                                                current_figure,
                                                False)

    labels_np = np.array(labels_new)

    labels_np[:,2] += 1
    condition = np.logical_or(labels_np[:,2] == 10, labels_np[:,2] == 26)
    labels_cat = np.where(condition, labels_np[:,0] * labels_np[:,1] * labels_np[:,2], labels_np[:,0] * labels_np[:,2])

    pth = label_save_path + '/' + 'sequence_labels'

    with open(pth, "bx") as fd:
        pickle.dump(labels_cat, fd)


if __name__ == '__main__':
    sequence_data()