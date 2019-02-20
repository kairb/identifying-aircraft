# from data_parser import Data
#
# training_images, training_labels = Data.create_hog_data_set(5)
# print(training_images.shape)
# print(training_labels.shape)


from image_parser import Parser
from histogram_of_gradients import HOG
from matplotlib import pyplot as plt

image = Parser.load_airport(1)
image1 = Parser.resize_image(image, 100)
image2 = Parser.resize_image(image, 200)
image3 = Parser.resize_image(image, 300)
print(len(image))
print(len(image1))
print(len(image2))
print(len(image3))