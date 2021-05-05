import numpy as np
import random


class PreProc:
    def __init__(self, scale=True, rmbkg=True,
                 resize=True):
        # This preproccessor class is intended to be used for training
        # The basic functionality of this class includes: capturing images,
        # image pre-proccessing, mutating images, and saving as a pickle

        # some pre-proccessing flags to determine the level of pre-proccessing
        # preformed to captured images
        self.f_scale = scale
        self.f_rm_background = rmbkg
        self.f_resize = resize

    def preproccess(self, image):
        # Alright, based on the flags, proccess the image stored in self.image

        self.image = image

        # remove the background (must be done before rescaling)
        if self.f_rm_background:
            self.remove_background()

        # rescale the image to be in the range of 0 to 1
        if self.f_scale:
            self.scale()

        if self.f_resize:
            self.resize()

        return self.image

    def remove_background(self):
        # First, find the minimum value in the depth image
        # for the depth image, the closer the object is to the camera,
        #  the smaller
        # the value. Thus, we want to find the closest pixel
        minVal = np.min(self.image[np.nonzero(self.image)])

        # Now we want to mask off the background. We do this by
        # setting any pixel thats further away than 1500 units from
        # the closest pixel to zero
        self.image[self.image > 1500 + minVal] = 0.

    def scale(self):
        # After masking off the background, find the furthest distance
        # in our ROI
        # (note, this could be minVal + 1500, but it could be smaller)
        maxVal = np.max(self.image[np.nonzero(self.image)])

        # Now we preform two operations at once, the first is to scale the ROI
        # relitive to itself
        # as such, the closest pixel should be near zero, and the furthest
        # near one.
        # Second, we raise this value to the fourth power, this is to help
        # make minor differences)
        # between pixels more distinct.
        # For example: the sign 'A' versus the sign 'S'
        # 'A' has the thumb closer to the camera than in 'S', but the
        # difference is thousanths of units. Thus, by raising to the fourth
        # power, we can increase that difference
        self.image = (self.image / maxVal) ** 4

        # Any small value (arbitrarily defined as smaller than 0.001,
        #  typically in the range ~E-5) is the
        # result of floating point errors with the masked off pixels
        # (I beleive)
        # Set these background pizels as 1, the furthest away in our range
        self.image[self.image < 0.001] = 1

    def resize(self, new_shape=(48, 64), operation='mean', image=None):
        """
        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        """
        if (image is None):
            ndarray = self.image
        else:
            ndarray = image

        operation = operation.lower()
        if operation not in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                             new_shape))
        compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                             ndarray.shape)]
        flattened = [thing for p in compression_pairs for thing in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))

        self.image = ndarray

        return ndarray


class ImageMan:
    def __init__(self, thresh=0.7):
        self.thresh = thresh

        self.sadface = np.array([
            [0., 1., 0., 1., 0.],
            [0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 0., 1., 0.],
        ])

    def threshold(self, image):
        # covert the image to a binary repersentation, with a threshold cut-off
        image_shape = np.shape(image)
        binary = np.ones(image_shape)

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if image[i][j] < self.thresh:
                    binary[i][j] = 0

        return binary

    def invert(self, image):
        # 1-image[i][j], seems to make the NN work better
        image_shape = np.shape(image)
        return np.array([[1-image[i][j] for j in range(image_shape[1])] for i
                        in range(image_shape[0])])

    def getBoundingVertex(self, image):
        # get the four corners of the bounding box for the ROI of an image
        a = np.sum(image, axis=0)
        b = np.sum(image, axis=1)

        # points = [0, image_shape[1], 0, image_shape[0]]
        points = np.zeros(4, dtype=np.int8)

        for i in range(0, int(len(a))):
            if(a[i] >= 3):
                points[0] = i
                break

        for i in range(len(a)-1, points[0], -1):
            if(a[i] >= 3):
                points[1] = i
                break

        for i in range(0, int(len(b))):
            if(b[i] >= 3):
                points[2] = i
                break

        for i in range(len(b)-1, points[2], -1):
            if(b[i] >= 3):
                points[3] = i
                break

        return points

    def getBounded(self, image):
        # get a binary repersentation thats even more simplified than the
        # origonal
        binary = self.threshold(image)

        # get the bounding box vertexes
        points = self.getBoundingVertex(binary)

        bounded = np.array([[image[i][j] for j in range(points[0], points[1])]
                           for i in range(points[2], points[3])])

        # make sure we dont return nothing
        if (np.shape(bounded)[0] == 0 or np.shape(bounded)[1] == 0):
            return self.sadface

        # return the origonal bounded image
        return bounded


class Mutator:
    def __init__(self, MutationFactor=10, MutationType="random",
                 ImageSize=(48, 64)):
        # record the specified muation types
        self.factor = MutationFactor
        self.type = MutationType
        self.size = ImageSize
        self.center = (int(self.size[0]/2), int(self.size[1]/2))

    def AutoMutate(self, bounded):
        # mutate the passed image based on the saved parameters

        # get the reversed bounded as well
        revd = np.flip(bounded, axis=1)

        # setup the output array
        output = [] 

        for _ in range(self.factor):
            # get the offset
            offset = (random.randrange(-20, 20, 1),
                      random.randrange(-30, 30, 1))

            # generate the mutant using either the fliped or non-fliped bound
            # image
            output.append(self.Insert(offset, random.choice([bounded, revd])))

        return output

    def Insert(self, offset, bound):
        # insert the bounded image, shifted from center by offset

        # get the dimensions of the origonal, bounded image
        bound_dim = np.shape(bound)
        bound_cent = (int(bound_dim[0]/2), int(bound_dim[1]/2))

        # create the new image of setup size
        mutant = np.zeros(self.size)

        # make sure that the bound image will fit within the mutant image size
        # given the offset
        margin = (int(self.size[0] / 2) - bound_cent[0] - offset[0],
                  int(self.size[1] / 2) - bound_cent[1] - offset[1])

        # check for overlow in the margins
        if (margin[0] < 0):
            # need to fix the y axis offset
            offset[0] += margin[0]

        if (margin[1] < 0):
            # need to fix the x axis offset
            offset[1] += margin[1]

        # insert the bounded image into the mutant!
        for y_b in range(bound_dim[0]):
            # calculate the index opsition on the mutant image to put the pixel
            # this is because offset is from the center of the images
            y_m = self.center[0] - offset[0] - bound_cent[0] + y_b

            for x_b in range(bound_dim[1]):
                # calculate the index opsition on the mutant image to put
                #  the pixel
                # this is because offset is from the center of the images
                x_m = self.center[1] - offset[1] - bound_cent[1] + x_b

                # copy the bounded image in!
                mutant[y_m][x_m] = bound[y_b][x_b]

        # return the mutant
        return mutant
