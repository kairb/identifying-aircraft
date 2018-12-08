# import statements
from sklearn import svm
import matplotlib.pyplot as plt
from image_parser import Parser
import numpy as np
from data_parser import Data

clf = svm.OneClassSVM(kernel="rbf", gamma=0.1, nu=0.5)
data = Data()
TRAINING_SET_SIZE = 90
TEST_IMAGE = 93

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

