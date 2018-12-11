# import statements
from sklearn import svm
from image_parser import Parser
from data_parser import Data

TRAINING_SET_SIZE = 90
TEST_IMAGE_1 = 91
TEST_IMAGE_2 = 192

for gamma in range(1, 10):
    for c in range(1, 100):
        clf = svm.SVC(gamma=gamma * 0.1, C=c, kernel="rbf")
        data = Data()

        ##get training data
        data.new_training_set(TRAINING_SET_SIZE)
        training_set = data.get_training_set()
        training_labels = data.get_training_labels()
        test_image_1 = data.get_test_image(TEST_IMAGE_1)
        test_image_2 = data.get_test_image(TEST_IMAGE_2)

        ##fit training data
        clf.fit(training_set, training_labels)

        print("gamma: ", gamma * 0.1, "C: ", c, "[1] ==:", clf.predict(test_image_1))
        print("gamma: ", gamma * 0.1, "C: ", c, "[2] ==:", clf.predict(test_image_2))

print("end of test")
