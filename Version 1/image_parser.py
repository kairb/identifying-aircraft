import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


class Parser:
    @staticmethod
    def load_flat_image(image_number):
        image = cv.imread("../images/" + str(image_number) + ".png", 0)
        return image.flatten()

    @staticmethod
    def load_image(image_number):
        image = cv.imread("../images/" + str(image_number) + ".png", 0)
        return image

    @staticmethod
    def load_other_image(image_number):
        image = cv.imread("../Images400x400/" + str(image_number) + ".png", 0)
        return image

    @staticmethod
    def load_images(images_list):
        ##new numpy array here
        for image_number in images_list:
            image = cv.imread("../images/" + str(image_number) + ".png", 1)
