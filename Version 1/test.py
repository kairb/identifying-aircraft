from data_parser import Data

training_images, training_labels = Data.create_hog_data_set(5)
print(training_images.shape)
print(training_labels.shape)
