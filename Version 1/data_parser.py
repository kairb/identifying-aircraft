from image_parser import Parser
import numpy as np


class Data:
    def __int__(self):
        self.images = []
        self.labels = []

    def new_training_set(self, amount):
        temp_images = []
        temp_labels = []
        for i in range(1, amount + 1):
            temp_images.append(Parser.load_flat_image(i))
            temp_labels.append(["aircraft", ])

        for j in range(101, 101 + amount + 1):
            temp_images.append(Parser.load_flat_image(i))
            temp_labels.append(["no aircraft54", ])

        self.images = np.asarray(temp_images)
        self.labels = np.ravel(temp_labels)

    @staticmethod
    def get_test_image(image_number):
        temp = []
        temp.append(Parser.load_flat_image(image_number))
        return np.asarray(temp)

    def get_training_set(self):
        return self.images

    def get_training_labels(self):
        return self.labels
