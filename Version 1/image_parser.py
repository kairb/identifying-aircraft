import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random


class Parser:

    @staticmethod
    def load_image_from_path(path):
        return cv.imread(path)

    @staticmethod
    def load_flat_image(image_number):
        image = cv.imread("../images/" + str(image_number) + ".png", 0)
        return image.flatten()

    @staticmethod
    def load_full_size_flat_image(image_number):
        return cv.imread("../images400x400/" + str(image_number) + ".png", 0).flatten()

    @staticmethod
    def load_image(image_number):
        return cv.imread("../images/" + str(image_number) + ".png", 0)

    @staticmethod
    def load_full_size_image(image_number):
        return cv.imread("../Images400x400/" + str(image_number) + ".png", 0)

    @staticmethod
    def rotate_image(image):
        return np.rot90(image, random.randint(1, 4))

    @staticmethod
    def load_airport(image_number):
        return cv.imread("../Airports/" + str(image_number) + ".png", 0)

    @staticmethod
    def resize_image(image, x,y):
        return cv.resize(image, (x,y))


