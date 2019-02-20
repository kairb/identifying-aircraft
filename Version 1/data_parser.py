from image_parser import Parser
import numpy as np
from histogram_of_gradients import HOG


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
    def create_resized_hog_data_set(dimension):
        """
        Returns resized data for training
        :param dimension:
        :return: images, labels for trainig
        """
        temp_images = []
        temp_labels = []

        for i in range(1, 100 + 1):
            temp_images.append(
                HOG.create_hog_image(Parser.rotate_image(Parser.resize_image(Parser.load_image(i), dimension))))
            temp_labels.append(1)

        for j in range(100, 200 + 1):
            temp_images.append(
                HOG.create_hog_image(Parser.rotate_image(Parser.resize_image(Parser.load_image(j), dimension))))
            temp_labels.append(0)

        images = np.asarray(temp_images)
        labels = np.ravel(temp_labels)
        return images, labels

    @staticmethod
    def create_airport_hog_data_set(image_number, x_step, y_step, search_size):
        image = np.asarray(Parser.load_airport(image_number))
        subsections = []
        for x in range(0, len(image[0]) - search_size+1, x_step):
            for y in range(0, len(image) - search_size+1, y_step):
                sub_image = image[y:y + search_size, x: x + search_size]
                subsections.append(HOG.create_hog_image(sub_image))

        return subsections


#Data.create_airport_hog_data_set(Parser.load_airport(2), 10, 10, 50)
