import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
import os


class Parser:
    @staticmethod
    def load_flat_image(image_number):
        image = cv.imread("../images/" + str(image_number) + ".png", 0)
        return image.flatten()

    @staticmethod
    def load_full_size_flat_image(image_number):
        image = cv.imread("../images400x400/" + str(image_number) + ".png", 0)
        return image.flatten()

    @staticmethod
    def load_image(image_number):
        image = cv.imread("../images/" + str(image_number) + ".png", 0)
        return image

    @staticmethod
    def load_full_size_image(image_number):
        image = cv.imread("../Images400x400/" + str(image_number) + ".png", 0)
        return image

    @staticmethod
    def rotate_image(data):
        return np.rot90(data, random.randint(1, 4))

