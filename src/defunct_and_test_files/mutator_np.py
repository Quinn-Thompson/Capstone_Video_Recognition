import numpy as np
import random
import scipy.ndimage as libimage
import pickle5 as pickle
import cv2
import random
import os
import math

import shutil
from psutil import virtual_memory

# it's possible to do all thre operations without reassigning the index fields. However, it is crazy complex and would probably begin slowing the program down
# as the checks for existance outside of the box would begin to require a lot more math. This method would invole resizing first, then shifting, then rotating.
# or atleast, this is the easierst. doing ROtation, shifting then resizing would require rotating the shifting ranges, then resizing them.
# its easier to just reassign the index fields and it doesn't slow it down at all.

def split_into_yolo(image, bbox, image_resolution_x, image_resolution_y):
    box_count = 8
    anchor_box_splitx = image_resolution_x // box_count
    anchor_box_splity = image_resolution_y // box_count
    object_count = 3

    reorder_labels = np.zeros((box_count, box_count, object_count, 3))

    bbox_f = bbox.flatten()

    anchor_locations_x = np.floor((bbox_f[0::2])/anchor_box_splitx)
    anchor_locations_y = np.floor(((bbox_f[1::2])//anchor_box_splity)*box_count)
    #print(bbox)

    anchor_locations = anchor_locations_x + anchor_locations_y
    #print(anchor_locations)

    unique, counts = np.unique(anchor_locations, return_counts=True)
    duplicate_anchors = [[[i, anchor_pos] for i, anchor_pos in enumerate(anchor_locations) if u == anchor_pos] for u in unique]
    #for i in range(box_count):
    #    for j in range(box_count):
    #        for k in range(object_count):
    #            reorder_labels[i][j][k][1] = 
    for ni,diff_anchors in enumerate(duplicate_anchors):
        for i, dup_anch in enumerate(diff_anchors):

            grid_x = int(dup_anch[1]%box_count)
            grid_y = int(dup_anch[1]//box_count)

            # rounding errors can pull x and y above the limit
            if grid_x == box_count:
                grid_x = box_count-1
            if grid_y == box_count:
                grid_y = box_count-1

            x1 = bbox_f[dup_anch[0]*2]
            y1 = bbox_f[(dup_anch[0]*2)+1]

            if x1 > image_resolution_x-1:
                x1 = image_resolution_x-1
            if y1 > image_resolution_y-1:
                y1 = image_resolution_y-1

            x1_adj = (x1 % (image_resolution_x/box_count)) / ((image_resolution_x/box_count)-1)
            y1_adj = (y1 % (image_resolution_y/box_count)) / ((image_resolution_y/box_count)-1)
            data_column = np.concatenate(([1], [x1_adj], [y1_adj]))

            #print(data_column)
            #print(int(dup_anch[1]%8))
            reorder_labels[grid_y,grid_x,i]= data_column
            #print(reorder_labels)
 

    return reorder_labels.flatten()

def create_ranges(start, stop, num_steps):
    # create multiple ranges at the same time
    steps = (1.0/ (num_steps-1)) * (stop - start)
    return steps[:,None]*np.arange(num_steps) + start[:,None]

def indices_array(image, n, object_center):
    # create a array where every pixel is represented by it's x, y coordinate
    x_shifted = np.arange(0-object_center[0], np.shape(image)[0]-object_center[0])
    y_shifted = np.arange(0-object_center[1], np.shape(image)[1]-object_center[1])
    out = np.empty((n,n,2),dtype=int)
    out[:,:,1] = y_shifted[:,None]
    out[:,:,0] = x_shifted
    return out

def create_rotation_angles(object_center, image, label, theta):
    run_points = np.subtract(label[:,0], object_center[0])
    rise_points = np.subtract(label[:,1], object_center[1])

    label_mags = np.sqrt(np.square(run_points) + np.square(rise_points))
    # calculate the original angle of the object poin
    # we don't use tan as it breaks at 90 degrees (sin/cos, 1/0)
    angle_noyflip = np.arccos(np.divide(run_points, label_mags))
    original_angles = np.where(np.arcsin(np.divide(rise_points, label_mags)) > 0, angle_noyflip, 2*np.pi-angle_noyflip)

    # I canned the term "series wall" as the circle does not intercept a box
    # but instead a series of four vertical wall lines that are spaced away from the object origin
    # this way it doesn't need to be passed four times for four seeprate arccos and arcsin methods, but as a single matrix
    series_wall_array = np.array([np.shape(image)[0]-object_center[0]-1, np.shape(image)[1]-object_center[1]-1, object_center[0], object_center[1]])
    wall_rotate = np.array([0, np.pi/2, np.pi, (3*np.pi)/2])
    # check the condition, if there could possibly be an intersection, as if the magnitude
    # or radius in this case is perpindicular to the series wall and it intersects, there is a cone of non existence
    # for image rotation
    condition_nonexistance = label_mags>np.array(series_wall_array)[:, None]
    # solve for the absolute angle of non existence, where no rotation can occur
    # then we add the subtract original point angle, as the current condition of existance is oriented around 
    # theta = 0. If this was not done, the ranges of non existances would be statically aligned with
    # the box edges. 
    # arccos doesn't distinguish between the negative and positive y axises, but we can assume the same angle at both ends
    # as the intersection of the radius line is a vertical line.
    # this means we simply repeat the radian value and subtract it from 2 pi.
    # these actions can't be split as it needs to be done before the correct index order is shifted and also before none is added, but not be applied to 
    # undefined intersections
    # we then need to orient the wall series back into a box formation, so we add a increasing series of rotations to the 
    # angle. 
    oppossite_ov_hyp = (series_wall_array[:, None])/label_mags[None, :]
    absolute_nonexistance_ranges = np.where(condition_nonexistance, ([-np.arccos(oppossite_ov_hyp), np.arccos(oppossite_ov_hyp)] - original_angles) + wall_rotate[:, None], None)

    # I couldn't figure out how to np.where to a non-existant array as opposed to none, as it 
    # always failed to cast it into a new shape without, so we also have to remove none indeces and recenter the polar coordinates to one unit circle in length
    polar_nonexistance_ranges = np.array([[angle % (2*np.pi) for wall in side for angle in wall if angle != None] for side in absolute_nonexistance_ranges ]).T
    
    angle_rotate = theta * (np.pi / 180)
    # need a seperate case just in case the polar range goes from 2pi to 0
    # this could probably be done better, use create_ranges for all existant ranges as opposed to this
    # but it's already so fast and the points are so few it doesn't really matter
    if len(polar_nonexistance_ranges) > 0 and polar_nonexistance_ranges[-1,1] < polar_nonexistance_ranges[-1,0]:
        left_over = polar_nonexistance_ranges[-1,1]
        polar_nonexistance_ranges[-1,1] = 2*np.pi
        setup_for_union = np.concatenate(([[0, left_over]], polar_nonexistance_ranges), axis=0).tolist()
    else:
        setup_for_union = np.concatenate(([[0, 0]], polar_nonexistance_ranges, [[2*np.pi, 2*np.pi]]), axis=0).tolist()

    # union each of the ranges
    unioned_ranges = []
    for begin, end in sorted(setup_for_union):
        if unioned_ranges and unioned_ranges[-1][1] >= begin:
            unioned_ranges[-1][1] = max(unioned_ranges[-1][1], end)
        else:
            unioned_ranges.append([begin, end])

    # read the areas betwee nthe non existance ranges and then stack it together. Bam
    allowed_angles = np.hstack([np.r_[unioned_ranges[i-1][1]:unioned_ranges[i][0]:angle_rotate] for i, _ in enumerate(unioned_ranges) if i != 0] )
    return allowed_angles


def rotate_image(image, index_region, object_center, label, range_of_rotations):

    # center label
    centered_label = np.subtract(label, object_center)
    # define rotation matrix 
    rotation_matrix_label = np.array([[np.cos(-range_of_rotations), np.sin(-range_of_rotations)],[-np.sin(-range_of_rotations), np.cos(-range_of_rotations)]])
    rotation_matrix = np.array([[np.cos(range_of_rotations), np.sin(range_of_rotations)],[-np.sin(range_of_rotations), np.cos(range_of_rotations)]])
    # dot product the rotation matrix and the shifted image indeces, then shift it back to the original origin, clip anything unbounded
    # so that the indeces repeat the nearest image edge
    rotated_index = np.asarray(np.clip(np.add(np.moveaxis(np.tensordot(index_region, rotation_matrix, axes=([2], [1])), -1, 0), object_center), 0, np.shape(image)[0]-1), dtype=np.int16)
    
    # reassign the color locations to new rotated indeces
    # why not just keep the rotation index and pass it to shift?
    # well, iamgine a iamge index where the entire thing is shifted 90 degrees
    # if you visualize this as a gradient from white (highest x) to black (lowest x)
    # it will be a gradient that will change from top to bottom
    # this p rogram shifts by adding a static value to each indece. If we add, lets say
    # 20 to each indece, then instead of shifting 20 to the right
    # it will shift by 20 DOWN. We need to reset the indeces.
    rotated_images = image[rotated_index[:,:,:,1], rotated_index[:,:,:,0]]
    
    # why do we not need to reassign these indeces?
    # they aren't representations of another static image
    # they are raw coordinates
    rotated_label = np.moveaxis(np.add(np.tensordot(centered_label, rotation_matrix_label, axes=([1], [1])), object_center[None, :, None]), -1, 0)



    return rotated_images, rotated_label

def shift_image(image, image_index, label, num_shifts):
    height = np.shape(image_index)[0]-1
    width = np.shape(image_index)[1]-1

    # create every possible shift location by tiling the x (1, 2, 3, 1, 2, 3, 1, 2, 3)
    # and repeatings the y                                 (1, 1, 1, 2, 2, 2, 3, 3, 3)
    x_rep = np.tile(np.asarray(create_ranges(np.min(label[:,:,0], axis=1), -(width - np.max(label[:,:,0], axis=1)), num_shifts), dtype=np.int16), num_shifts)
    y_rep = np.repeat(np.asarray(create_ranges(np.min(label[:,:,1], axis=1), -(height - np.max(label[:,:,1], axis=1)), num_shifts), dtype=np.int16), num_shifts, axis=1)
    # create a line for every possible index of the rotation image
    image_rep = np.arange(0, len(image))
    # concatenate the two different x and y shifts for an easier adding operation
    shift_indeces = np.concatenate((x_rep[:,:,None], y_rep[:,:,None]), axis=2) 
    # add every possible shift to the image, round it so it can be accessed as an index, clamp it at 0-image_len and swap the first and the fourth dimension
    # so it can be parsed for each of the different image rotations and each of the index shifts
    shifted_indeces = np.moveaxis(np.clip(np.asarray(np.add(image_index[None, None, :, :], shift_indeces[:, :, None, None, :]), dtype=np.int16), 0, np.shape(image_index)[1]-1), 0, -2)

    # grab the color values for each of the rotated images at the shifted indece, the nswap the dimensions back
    image_shift = np.moveaxis(image[image_rep, shifted_indeces[:, :, :, :, 1], shifted_indeces[:, :, :, :, 0]], -2, 0)
    # shift the label, but in the negative, same as how we use a negative theta to rotate, as it's represented by image x and y values (up and right), as opposed to
    # array x and y (down and right)
    label_shift = np.add(label[:, None, :, :], -shift_indeces[:, :, None, :])

    # flatten it so the math is easier later on
    flatten_image = np.reshape(image_shift, (np.shape(image_shift)[0]*np.shape(image_shift)[1], *np.shape(image_shift)[2:]))
    flatten_label = np.reshape(label_shift, (np.shape(label_shift)[0]*np.shape(label_shift)[1], *np.shape(label_shift)[2:]))

    return flatten_image, flatten_label

def resize_image(image, image_index, label, num_resizes, object_center, range_minimum, range_maximum):

    # center label, in this case doubles for box size and centering the resize
    centered_label = np.subtract(label, object_center[:, None])

    # create a line for every possible index of the images
    image_rep = np.arange(0, len(image))

    box_size = np.vstack((np.min( (centered_label[:,:,0]), axis=1), np.min( (centered_label[:,:,1]), axis=1), 
                         np.max( (centered_label[:,:,0]), axis=1), np.max( (centered_label[:,:,1]), axis=1))).T
    # calculates the lengths from the center of the object 
    center_to_edge = np.hstack((object_center, np.shape(image_index)[1]-object_center))

    #calculates the max possible resize amounts
    resize_max = np.min(np.concatenate((np.abs(center_to_edge / box_size), np.repeat(range_maximum, len(object_center))[:, None]), axis=1), axis=1)

    # creates a range based on the min input and max input (or calculated max for object to still exist in frame)
    range_resize = create_ranges(np.repeat(range_minimum, len(resize_max)), resize_max, num_resizes)

    # resizes every index of the image based on the object center
    resized_indeces = np.swapaxes(np.asarray(np.clip(np.add(np.multiply(image_index[:, :, :, None], range_resize[:, None, None, :, None]), object_center[:, None, None, None, :]), 
                      0, np.shape(image_index)[1]-1), dtype=np.int16), -2, 0)

    # map it to the rotated images
    image_resized = np.moveaxis(image[image_rep, resized_indeces[:, :, :, :, 1], resized_indeces[:, :, :, :, 0]], -2, 0)

    # resize the labels
    label_resized = np.add(np.divide(centered_label[:, None, :, :], range_resize[:, :, None, None]), object_center[:, None, None, :])

    # flatten for easier math
    flatten_image = np.reshape(image_resized, (np.shape(image_resized)[0]*np.shape(image_resized)[1], *np.shape(image_resized)[2:]))
    flatten_label = np.reshape(label_resized, (np.shape(label_resized)[0]*np.shape(label_resized)[1], *np.shape(label_resized)[2:]))

    return flatten_image, np.asarray(flatten_label, dtype=np.int16)

def soft_bleed(image, image_index, blur_alpha):
    near_indeces_slide = np.vstack(([1, 0], [0, 1], [-1, 0], [0, -1]))

    image_rep = np.arange(0, len(image))

    scuff_indeces = np.clip(np.add(image_index, near_indeces_slide[:, None, None, :]), 0, np.shape(image)[1]-1)
    scuff_images = image[:, scuff_indeces[:, :, :, 1], scuff_indeces[:, :, :, 0]] * ((1-blur_alpha)/4)

    blurred_images =  np.sum(np.concatenate((scuff_images, image[:, None]*blur_alpha), axis=1), axis=1)

    return np.asarray(blurred_images, dtype=np.uint8)
 

def edit_image():
    path = 'database\\figure_wo_shift\\'
    label_path = 'database\\label_wo_shift\\'
    new_path = 'database\\figure_w_shift\\'
    new_label_path = 'database\\label_w_shift\\'
    if os.path.isdir(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)
    if os.path.isdir(new_label_path):
        shutil.rmtree(new_label_path)
    os.makedirs(new_label_path)

    resize_og_image = 2

    num_shifts = 4
    theta = 20
    num_resizes = 3
    range_resize_max = 1.0
    range_resize_min = 0.7

    # percent amount retained form the original pixel
    blur_alpha = 0.2

    with open(label_path + 'label_wo_shift', "rb") as fd:
        labels = pickle.load(fd)

    file_names = os.listdir(path)

    iterator = 0

    for i, _file in enumerate(file_names):
        empty = False
        with open(path + _file, "rb") as fd:
            image = pickle.load(fd)
        label = np.asarray(labels[int(_file)]) // resize_og_image

        image = cv2.resize(image, (np.shape(image)[0] // resize_og_image, np.shape(image)[1] // resize_og_image))
        # conserves memory to a maximum at the developers discretion

 

        # rotate
        # assume chunking is not needed
        # the object center can be calculated using any method, as all the calculations should be relative to it
        # ae, the image will rotate around the object center regardless of where it is
        # and the image will be resized based on the object center
        object_center = np.empty(2)
        object_center[0] = (min(label[:,0])+max(label[:,0])) * 0.5
        object_center[1] = (min(label[:,1])+max(label[:,1])) * 0.5
        range_of_rotations = create_rotation_angles(object_center, image, label, theta)
        index_region = indices_array(image, np.shape(image)[0], object_center)
        rotated_images, rotated_labels = rotate_image(image, index_region, object_center, label, range_of_rotations)

        #mcs_atores = int(virtual_memory().free / (((len(rotated_images) * num_resizes) * (np.shape(rotated_images)[1])**2 * float_size) * 2 * 3))

        # resize
        #resize_chunking = np.arange(0, len(rotated_images), mcs_atores)
        #resize_chunking = np.concatenate((resize_chunking, [len(rotated_images)]))
        #catchunk_resimages = np.array([]).reshape(0, *np.shape(rotated_images)[1:])
        #catchunk_reslabels = np.array([]).reshape(0, *np.shape(rotated_labels)[1:])
        # why do we chunk out these operations?
        # well, because if we allow the range to get too large it will begin to page to virtual memory
        # this is ~ 1000-1 million times slower than just iterating through smaller chunks
        # as it requires parsing large segments of memory from the hard drive
        #for j, ce in enumerate(resize_chunking[1:]):
        #    cb = resize_chunking[j]
        rotated_object_centers = np.empty((len(rotated_images), 2))
        rotated_object_centers[:, 0] = np.min(rotated_labels[:,:,0], axis = 1) + np.max(rotated_labels[:,:,0], axis = 1) * 0.5
        rotated_object_centers[:, 1] = np.min(rotated_labels[:,:,1], axis = 1) + np.max(rotated_labels[:,:,1], axis = 1) * 0.5
        index_reassigned_region = indices_array(image, np.shape(image)[0], [0, 0]) - rotated_object_centers[:, None, None, :]
        resized_images, resized_labels = resize_image(rotated_images, index_reassigned_region, rotated_labels, num_resizes, rotated_object_centers, range_resize_min, range_resize_max)
        #catchunk_resimages = np.vstack([catchunk_resimages, resized_images]) 
        #catchunk_reslabels = np.vstack([catchunk_reslabels, resized_labels])

        # shift
        index_reassigned_region = indices_array(image, np.shape(image)[0], [0, 0])
        shifted_images, shifted_labels = shift_image(resized_images, index_reassigned_region, resized_labels, num_shifts)

        # blur
        # why blur? this is a very basic blur that basically takes a portion of the original pixel (blur_alpha)
        # and adds it to the four pixels that are directly adjacent to it
        # the issue with our method of rotation is that it causes the stairstepping issue with straight lines
        # as there is no interpolation method (well, there is no direct interpolation calculation, but the de facto interpolation  
        # is basically piecewise constant/nearest neighbor interpolation as it interprets nearest indeces) the blur removes this artifact somewhat, but it's not entirely necessary

        blurred_images = soft_bleed(shifted_images, index_reassigned_region, blur_alpha)

        # this is then split into you only look once format
        for i, image_k in enumerate(blurred_images):
            output_label = split_into_yolo(image_k, shifted_labels[i], np.shape(image_k)[1], np.shape(image_k)[0])
            pth = new_path + str(i+iterator).zfill(7)
            with open(pth, "bx") as fd:                    
                pickle.dump(image_k, fd)
            pth = new_label_path + str(i+iterator).zfill(7)
            with open(pth, "bx") as fd:
                pickle.dump(output_label, fd)
        iterator += i + 1



if __name__ == '__main__':
    edit_image()