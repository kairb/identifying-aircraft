# import statements
from sklearn import svm
import matplotlib.pyplot as plt
from image_parser import Parser
from data_parser import Data

# declaring data set paramaters
TRAINING_SET_SIZE = 90
TEST_IMAGES = [95, 195, 96, 97, 198]

# get training data in the form of feature descriptors
training_set, training_labels = Data.create_hog_data_set(TRAINING_SET_SIZE)
test_images = Data.create_hog_image(TEST_IMAGES)

clf = svm.SVC(gamma=0.0001, C=10, probability=True)

# fit training data
clf.fit(training_set, training_labels)
# print(clf.predict_proba(training_set))

result = clf.predict(test_images)
print("Prediction: ", clf.predict(test_images))

MAX_COLUMNS = 5
MAX_ROWS = len(result)/MAX_COLUMNS




plt.imshow(Parser.load_image(TEST_IMAGES[0]))

plt.show()
