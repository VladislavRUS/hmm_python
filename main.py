import numpy as np
from hmmlearn import hmm
import scipy.ndimage


def split_horizontally(images, images_number):
    result = []

    for image in images:
        center = scipy.ndimage.measurements.center_of_mass(image)
        y = int(center[0])
        height = len(image)
        top = image[:, ::y]
        bottom = image[:, y::]

        result.append(top)
        result.append(bottom)

    if len(result) != images_number:
        images = result

    else:
        split_horizontally(result, images_number)


# def split(image, split_type):


def start():
    base_dir = './TrainingSet'
    offline_forgeries = 'Offline Forgeries'
    offline_genuine = 'Offline Genuine'

    test_image = '001_01.PNG'

    img = scipy.misc.imread(base_dir + '/' + offline_genuine + '/' + test_image)

    images = [img]

    print(images)
    split_horizontally(images, 2)

