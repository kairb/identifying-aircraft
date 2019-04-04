import numpy as np
from PIL import Image, ImageDraw
from image_parser import Parser


import numpy as np
import cv2 as cv


class Overlay:
    image = None

    def __init__(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def draw_boxes(self, probabilities, size_x,size_y,x_step,y_step):

        i = 0
        for x in range(0, len(self.image[0]) - size_x + 1, x_step):
            for y in range(0, len(img) - size_y + 1, y_step):
                if probabilities[i][1] > 0.4:
                    img = cv.rectangle(self.image, (x,y), (x+size_x, y + size_y))
                    draw.rectangle([x, y, size_x, size_y], outline="red")
                i += 1

        return img





class Draw:


    @staticmethod
    def draw_predictions(predictions, image_path, size_x, size_y, x_step, y_step):
        """
        draws boxes around predictions
        :param predictions: list of predictions
        :param image_path: path of image to draw predictions om
        :return: image with predictions on as np array
        """
        img = Image.fromarray(Parser.load_image_from_path(image_path))
        draw = ImageDraw.Draw(img)
        i = 0
        for x in range(0, len(img[0]) - size_x + 1, x_step):
            for y in range(0, len(img) - size_y + 1, y_step):
                if predictions[i] == 1:
                    draw.rectangle([x, y, size_x, size_y], outline="red")
                i += 1

        return img

Draw.draw_predictions()
# https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html
