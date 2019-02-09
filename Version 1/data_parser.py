from image_parser import Parser
import numpy as np
from HistogramOfGradients import HOG


class Data:
    @staticmethod
    def new_training_set(amount):
        temp_images = []
        temp_labels = []

        for j in range(101, 101 + amount + 1):
            temp_images.append(Parser.load_flat_image(j))
            temp_labels.append(0)

        for i in range(1, amount + 1):
            temp_images.append(Parser.load_flat_image(i))
            temp_labels.append(1)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def new_full_size_training_set(amount):
        temp_images = []
        temp_labels = []

        for j in range(101, 101 + amount + 1):
            temp_images.append(Parser.load_full_size_flat_image(j))
            temp_labels.append(0)

        for i in range(1, amount + 1):
            temp_images.append(Parser.load_full_size_flat_image(i))
            temp_labels.append(1)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def get_test_image(image_number):
        temp = []
        temp.append(Parser.load_full_size_flat_image(image_number))
        return np.asarray(temp)

    @staticmethod
    def create_hog_test_set(image_number):
        temp = []
        for i in image_number:
            temp.append(HOG.create_hog_image(Parser.load_full_size_image(i)))
        return np.asarray(temp)

    @staticmethod
    def create_hog_data_set(amount):
        temp_images = []
        temp_labels = []

        for i in range(1, amount + 1):
            temp_images.append(HOG.create_hog_image(Parser.load_full_size_image(i)))
            temp_labels.append(1)

        for j in range(100, 100 + amount + 1):
            temp_images.append(HOG.create_hog_image(Parser.load_full_size_image(j)))
            temp_labels.append(0)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def create_realistic_hog_data_set(amount):
        temp_images = []
        temp_labels = []

        for i in range(1, amount + 1):
            temp_images.append(HOG.create_hog_image(Parser.rotate_image(Parser.load_full_size_image(i))))
            temp_labels.append(1)

        for j in range(100, 100 + amount + 1):
            temp_images.append(HOG.create_hog_image(Parser.rotate_image(Parser.load_full_size_image(j))))
            temp_labels.append(0)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def create_realistic_hog_test_set(image_number):
        temp = []
        for i in image_number:
            temp.append(HOG.create_hog_image(Parser.rotate_image(Parser.load_full_size_image(i))))
        return np.asarray(temp)

    #creates two feature descriptor arrays
    @staticmethod
    def create_fd_arrays():
        afds = []
        gfds = []
        for i in range(1, 101):
            afds.append(HOG.create_hog_image(Parser.load_full_size_image(i)))
        for j in range(101, 201):
            gfds.append(HOG.create_hog_image(Parser.load_full_size_image(j)))

        return afds, gfds
