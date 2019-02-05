# import statements
from sklearn import svm
import matplotlib.pyplot as plt
from image_parser import Parser
from data_parser import Data

# declaring data set paramaters
TRAINING_SET_SIZE = 90
TEST_IMAGES = [95, 195, 96, 97, 198, 199, 98, 99, 88, 191]

# get training data in the form of feature descriptors
training_set, training_labels = Data.create_hog_data_set(TRAINING_SET_SIZE)
test_images = Data.create_hog_image(TEST_IMAGES)
# GUI
MAX_COLUMNS = 5
MAX_ROWS = 2

clf = svm.SVC(gamma=0.0001, C=10)

# fit training data
clf.fit(training_set, training_labels)

result = clf.predict(test_images)
print("Prediction: ", result)

for i in range(1, len(test_images) + 1):
    plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
    plt.imshow(Parser.load_full_size_image(TEST_IMAGES[i - 1]), cmap="gray")
    if result[i - 1] == 1:
        plt.title("Aircraft")
    else:
        plt.title("Ground")
    plt.axis('off')

plt.show()
