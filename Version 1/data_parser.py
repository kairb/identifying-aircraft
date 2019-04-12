from image_parser import Parser
import numpy as np
from histogram_of_gradients import HOG
import os


class Data:

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

    # creates two feature descriptor arrays
    @staticmethod
    def create_fd_arrays():
        afds = []
        gfds = []
        for i in range(1, 101):
            afds.append(HOG.create_hog_image(Parser.load_full_size_image(i)))
        for j in range(101, 201):
            gfds.append(HOG.create_hog_image(Parser.load_full_size_image(j)))

        return afds, gfds

    @staticmethod
    def create_resized_hog_data_set(x, y):
        """
        Returns resized data for training using old data set
        :param dimension:
        :return: images, labels for trainig
        """
        temp_images = []
        temp_labels = []

        for i in range(1, 100 + 1):
            temp_images.append(
                HOG.create_hog_image(Parser.rotate_image(Parser.resize_image(Parser.load_image(i), x, y))))
            temp_labels.append(1)

        for j in range(100, 200 + 1):
            temp_images.append(
                HOG.create_hog_image(Parser.rotate_image(Parser.resize_image(Parser.load_image(j), x, y))))
            temp_labels.append(0)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def create_training_data(x, y):
        """
        Returns resized hog data for training using newly gen images
        :param x: size
        :param y: size
        :return: hog array, labels for training
        """
        temp_images = []
        temp_labels = []
        aircraft_path = "../Aircraft/"
        ground_path = "../Ground/"

        for image in os.listdir(ground_path):
            temp_images.append(
                HOG.create_hog_image(
                    Parser.rotate_image(Parser.resize_image(Parser.load_image_from_path(ground_path + image), x, y))))
            temp_labels.append(0)

        for image in os.listdir(aircraft_path):
            temp_images.append(
                HOG.create_hog_image(
                    Parser.rotate_image(Parser.resize_image(Parser.load_image_from_path(aircraft_path + image), x, y))))
            temp_labels.append(1)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def create_airport_hog_data_set(path, x_steps, y_steps, size_x, size_y):
        image = np.asarray(Parser.load_image_from_path(path))
        images = []
        hog_subsections = []
        x_step = int((len(image[0]) - size_x) / x_steps)
        y_step = int((len(image) - size_y) / y_steps)

        for x in range(0, len(image[0]) - size_x + 1, x_step):
            for y in range(0, len(image) - size_y + 1, y_step):
                sub_image = image[y:y + size_y, x: x + size_x]
                images.append(sub_image)
                hog_subsections.append(HOG.create_hog_image(sub_image))

        return hog_subsections, images
