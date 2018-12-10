# import statements
from sklearn import svm
import matplotlib.pyplot as plt
from image_parser import Parser
import numpy as np
from data_parser import Data

TRAINING_SET_SIZE = 90
TEST_IMAGE = 93

clf = svm.SVC(gamma=0.001, C=100)
data = Data()

##get training data
data.new_training_set(TRAINING_SET_SIZE)
training_set = data.get_training_set()
training_labels = data.get_training_labels()
test_image = data.get_test_image(TEST_IMAGE)


##fit training data
clf.fit(training_set, training_labels)

print("Prediction: ", clf.predict(test_image))

plt.imshow(Parser.load_image(TEST_IMAGE))

plt.show()
