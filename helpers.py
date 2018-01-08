import os
import warnings

import numpy as np
import scipy.cluster
import scipy.fftpack
import scipy.ndimage
import skimage.transform
from hmmlearn import hmm
from sklearn.externals import joblib

warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_vertically(images, images_number):
    result = []

    for image in images:
        height = image.shape[0]
        width = image.shape[1]

        center = scipy.ndimage.measurements.center_of_mass(image)
        center_x = int(center[1])
        left = image[0:height, 0:center_x]
        right = image[0:height, center_x:width]

        result.append(left)
        result.append(right)

    if len(result) == images_number:
        return result

    else:
        return split_vertically(result, images_number)


def split_horizontally(images, images_number):
    result = []

    for image in images:
        height = image.shape[0]
        width = image.shape[1]

        center = scipy.ndimage.measurements.center_of_mass(image)
        center_y = int(center[0])
        top = image[0:center_y, 0:width]
        bottom = image[center_y:height, 0:width]

        result.append(top)
        result.append(bottom)

    if len(result) == images_number:
        return result

    else:
        return split_horizontally(result, images_number)


def split_mixed(images, depth, split_type):
    result = []

    if split_type == 'hor':

        result = split_horizontally(images, len(images) * 2)
        split_type = 'ver'

    elif split_type == 'ver':
        result = split_vertically(images, len(images) * 2)
        split_type = 'hor'

    if len(result) == depth ** 2:
        return result

    else:
        return split_mixed(result, depth, split_type)


def to_binary(image):
    mean = image.mean()
    height = image.shape[0]
    width = image.shape[1]

    binary = []

    for i in range(0, height):

        row = []
        for j in range(0, width):
            if image[i, j] > mean:
                row.append(255)

            else:
                row.append(0)

        binary.append(np.array(row))

    return np.array(binary)


def get_training_data(train_features):
    result = []
    lengths = []

    for vector in train_features:
        row = []
        lengths.append(len(vector))

        for value in vector:
            row.append(np.array(value))

        result.append(np.array(row))

    result = np.concatenate(result)

    return result, lengths


def get_image_features(image_name):
    signature_img = scipy.ndimage.imread(image_name, flatten=True)
    signature_img = scipy.ndimage.filters.gaussian_filter(signature_img, 2)
    signature_img = scipy.misc.imresize(signature_img, 0.5)
    signature_img = to_binary(signature_img)

    initial_parts = split_vertically([signature_img], 4)

    result = []

    for i in range(0, len(initial_parts)):
        part_state = initial_parts[i]
        observations = split_mixed([part_state], 4, 'hor')

        for observation in observations:
            dct_observation = scipy.fftpack.dct(observation.flatten())[0:64]
            result.append(dct_observation)

    return result


def extract_image_name(file_name):
    return file_name[file_name.rfind('/') + 1::].split('_')[0]
