from image_parser import Parser
import numpy as np


class Data:

    def new_training_set(self, amount):
        temp_images = []
        temp_labels = []

        for j in range(101, 101 + amount + 1):
            temp_images.append(Parser.load_flat_image(j))
            temp_labels.append(["no aircraft", ])

        for i in range(1, amount + 1):
            temp_images.append(Parser.load_flat_image(i))
            temp_labels.append(["aircraft", ])

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def get_test_image(image_number):
        temp = []
        temp.append(Parser.load_flat_image(image_number))
        return np.asarray(temp)
