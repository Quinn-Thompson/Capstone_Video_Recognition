import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.RSC_Wrapper import RSC
from tools.preproc import PreProc
import numpy as np
import cv2

import pickle5 as pickle
import os

def draw_bb(event, x, y, latchesl, coordsl, color):
    global display_drawn_on, display
    if event == cv2.EVENT_MOUSEMOVE:
        display_drawn_on = display.copy()
        cv2.line(display_drawn_on, (x, 0), (x, 511), color, 2)
        cv2.line(display_drawn_on, (0, y), (511, y), color, 2)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if latchesl[0] == False:
            coordsl[0] = x//8
            coordsl[1] = y//8
            latchesl[0] = True
        else:
            coordsl[2] = x//8
            coordsl[3] = y//8
            latchesl[1] = True
            display = display_drawn_on
    if event == cv2.EVENT_MOUSEMOVE and latchesl[0] == True:
        display_drawn_on = display.copy()
        cv2.rectangle(display_drawn_on, (int(coordsl[0]*8), int(coordsl[1]*8)), (x, y), color, 2)
    return latchesl, coordsl

def draw_rect(event,x,y,flags,param):
    
    global coords
    global latches
    
    #print(latches)
    for i in range(num_obj_detect):
        if i == 0:
            if latches[1] == False:
                color = 0.5
                latches[0:2], coords[0:4] = draw_bb(event, x, y, latches[0:2], coords[0:4], color)
                break
        else:
            if latches[(i*2)-1] == True and latches[(i*2)+1] == False:
                color = 0.5
                latches[2*i:2*(i+1)], coords[4*i:4*(i+1)] = draw_bb(event, x, y, latches[2*i:2*(i+1)], coords[4*i:4*(i+1)], color)
                break
            
        
    
    
def run_stream():

    # this is a poor method of extracting information
    # it needs to be re-written or handled in the gui
    # what are the gesture confidence and video confidence?
    # read white paper, it's a pretty non-intuitive idea
    # this is also leftover and has not been removed yet
    # as the confidence idea can be used for networks with
    # and without confidence neurons

    global num_obj_detect
    num_obj_detect = 1
    
    num_coords = 4 * num_obj_detect
    num_latches = 2 * num_obj_detect

    global checked
    global display
    global display_drawn_on
    # face*blue*, hand on left (right hand)*red*, hand on right (left_hand)*green*
    global coords
    coords = [0] * num_coords
    global latches
    latches = [False] * num_latches

    stream_length = 24

    files_training = os.listdir("exports/")
    files_post_given = os.listdir("database/gestures_seperate/")


    dif = np.setdiff1d(files_training,files_post_given)
    for training_data in dif:
        with open("exports/" + training_data, "rb") as fd:
            image_stream = pickle.load(fd)
        bbox = np.zeros(shape=(len(image_stream),4))
        labels = np.zeros(shape=(len(image_stream),3))

        print('enter your letter')
        letter = cv2.waitKey(0) - 97
        print(letter)

        cv2.destroyAllWindows()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rect)
        current_folders = len(os.listdir("./database/" + "gestures_seperate/"))

        os.mkdir(path = "./database/" + "gestures_seperate/" + training_data)
        os.mkdir(path = "./database/" + "bbox_seperate/" + training_data)
        os.mkdir(path = "./database/" + "labels_seperate/" + training_data)

        deleted_frames = 0

        # run through each of the images taken during the streaming proces
        for i, image in enumerate(image_stream):
            # print the current image iteration so the dev knows to delete it if theyu mess up
            print(i)
            # copy the image
            display = cv2.resize(image.copy(), (512,384))
            display_drawn_on = cv2.resize(image.copy(), (512,384))
            
            latches[:] = [False]*(num_latches)

            # while the image has not been checked off
            cv2.imshow('image', display_drawn_on)
            cv2.moveWindow('image',10,10)
            for j in range(8):
                if i+j < stream_length:
                    cv2.imshow('image' + str(j), image_stream[i+j])
                    cv2.moveWindow('image' + str(j),10+(j*100),500)

            print('enter command (5 == delete image, 7 == replicate info, 9 == override letter)')
            command = cv2.waitKey(0) - 48


            if command != 5:
                if command != 7:
                    print('enter end gesture confidence')
                    confidence = cv2.waitKey(0) - 48
                    print('confidence is now: ' + str(confidence))
                    print('enter total gesture confidence')
                    gesture_confidence = cv2.waitKey(0) - 48
                    print('gesture confidence is now: ' + str(gesture_confidence))

                    #for j in range(num_obj_detect):
                    #    if category & pow(2, j) == 0:
                    #        coords[j*4:(j+1)*4] = [0.0] * 4
                    #        latches[j*2:(j+1)*2] = [True] * 2

                    while all(latches) == False:
                        cv2.imshow('image', display_drawn_on)
                        k = cv2.waitKey(20)


                pth = "./database/" + "gestures_seperate/" + str(letter) + str(current_folders).zfill(5) + '/' + str(i).zfill(5)
                with open(pth, "bx") as fd:                    
                    pickle.dump(image, fd)



                labels[i-deleted_frames][0] = confidence
                labels[i-deleted_frames][1] = gesture_confidence

                if command == 9:
                    print('enter override letter')
                    temp_letter = cv2.waitKey(0) - 97
                    print('override letter is now: ' + str(temp_letter))
                    labels[i-deleted_frames][2] = temp_letter
                else:
                    labels[i-deleted_frames][2] = letter

                bbox[i-deleted_frames] = coords

                print(bbox[i-deleted_frames])
                print(labels[i-deleted_frames])
            else:
                deleted_frames += 1

                    


        pth = "./database/" + "labels_seperate/" + str(letter) + str(current_folders).zfill(5) + "/labels"
        
        if os.path.isfile(pth):
            os.remove(pth)
        
        with open(pth, "bx") as fd:
            pickle.dump(np.array(labels), fd)

        pth = "./database/" + "bbox_seperate/" + str(letter) + str(current_folders).zfill(5) + "/bbox"

        if os.path.isfile(pth):
            os.remove(pth)

        with open(pth, "bx") as fd:
            pickle.dump(np.array(bbox), fd)

        
if __name__ == "__main__":
    run_stream()
