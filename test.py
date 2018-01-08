import os
import warnings

import numpy as np
import scipy.cluster
import scipy.fftpack
import scipy.ndimage
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

def extract_image_name(file_name):
    return file_name[file_name.rfind('/') + 1::].split('_')[0]

def get_image_features(image_name):
    signature_img = scipy.ndimage.imread(image_name, flatten=True)
    signature_img = scipy.ndimage.filters.gaussian_filter(signature_img, 0.5)

    initial_parts = split_vertically([signature_img], 4)

    result = []

    for i in range(0, len(initial_parts)):
        part_state = initial_parts[i]
        observations = split_mixed([part_state], 4, 'hor')

        for observation in observations:
            dct_observation = scipy.fftpack.dct(observation.flatten())[0:1]
            result.append(dct_observation)

    return result

def start():
    base_dir = './TrainingSet'
    offline_forgeries = 'Offline Forgeries'
    offline_genuine = 'Offline Genuine'
    models_dir = './models'

    model_files = os.listdir(models_dir)
    genuine_signatures = os.listdir(base_dir + '/' + offline_genuine)

    signatures_dictionary = {}

    for signature_file_name in genuine_signatures:
        key = signature_file_name.split('_')[0]

        if key not in signatures_dictionary:
            signatures_dictionary[key] = []

        path_to_signature = base_dir + '/' + offline_genuine + '/' + signature_file_name
        signatures_dictionary[key].append(path_to_signature)

    test_signatures = []

    for key in signatures_dictionary:
        signatures = signatures_dictionary[key][5:10]

        for signature in signatures:
            test_signatures.append(signature)


    errors = 0

    for signature in test_signatures:

        scores = []

        for model_file in model_files:
            hmm_model = joblib.load(models_dir + '/' + model_file)
            scores.append(hmm_model.score(get_image_features(signature)))

        highest_score_model = model_files[np.argmax(scores)]

        image_name = extract_image_name(signature)
        model_name = highest_score_model.split('.')[0]

        if image_name != model_name:
            errors = errors + 1

    print(1 - (errors / len(test_signatures)))

start()