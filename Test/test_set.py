import numpy as np


def create_test_set(amount):
    images = []
    labels = []
    for i in range(0, amount):
        images.append(np.zeros((10, 10)).flatten())
        labels.append(0)
        images.append(np.full((10, 10), 255).flatten())
        labels.append(1)

    return np.asarray(images), np.asarray(labels)


def create_white_image():
    return np.full((10, 10), 255).flatten()


def create_black_image():
    return (np.zeros((10, 10)).flatten())

