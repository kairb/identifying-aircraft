import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class Draw:
    image = None

    def __init__(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def draw_boxes(self, probabilities, size_x,size_y,x_step,y_step):

        """
        Draws boxes over image dependent on probabilities
        :param probabilities: List of probabilities
        :param size_x: size of search area
        :param size_y: size of search area
        :param x_step: x step of search areas
        :param y_step: y step of search area
        :return: numpy array. image with boxes showing aircraft locations
        """
        img = self.image
        i = 0
        for x in range(0, len(img[0]) - size_x + 1, x_step):
            for y in range(0, len(img) - size_y + 1, y_step):
                if probabilities[i][1] > 0.4:
                    img = cv.rectangle(img, (x,y), (x+size_x, y + size_y),(0,255,0),3)
                i += 1
        print("boxes", img.shape)
        return img

    def draw_colour_gradient(self,probabilities, size_x,size_y,x_step,y_step):
        blank = np.zeros((len(self.image), len(self.image[0])))
        print("blank", blank.shape)
        product = []
        i = 0
        for x in range(0, len(blank[0]) - size_x + 1, x_step):
            for y in range(0, len(blank) - size_y + 1, y_step):
                if probabilities[i][1] > 0.4:
                    blank = np.zeros((len(self.image), len(self.image[0])))
                    intensity = probabilities[i][1] * 255
                    temp = cv.rectangle(blank, (x, y), (x + size_x, y + size_y), intensity, -1)
                    product.append(temp.flatten())
                i += 1
        img = np.average(product,axis = 0)
        img = img.reshape((len(self.image), len(self.image[0])))
        print("average", img.shape)
        return img
