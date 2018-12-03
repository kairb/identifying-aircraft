# import statements
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
from image_parser import Parser
import numpy as np
from data_parser import Data

clf = svm.OneClassSVM()
data = Data()

##get training data
data.new_training_set(90)
training_set = data.get_training_set()
training_labels = data.get_training_labels()
test_image = data.get_test_image(102)

clf.fit(training_set, training_labels)

print("test image shape", test_image.shape)
print("Prediction: ", clf.predict(test_image))

plt.imshow(Parser.load_image(90))

plt.show()
