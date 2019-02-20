# import statements
from sklearn import svm
import matplotlib.pyplot as plt
from image_parser import Parser
from data_parser import Data

# declaring data set paramaters
IMAGE_SIZE = None
X_STEP = None
Y_STEP = None
CLASSIFICATION_METHOD = None

while True:
    CLASSIFICATION_METHOD = input("1 for normal classification\n"
                                  "2 for search based classification: ")
    if CLASSIFICATION_METHOD.isdigit():
        CLASSIFICATION_METHOD = int(CLASSIFICATION_METHOD)
        break



# normal classification
if CLASSIFICATION_METHOD == 1:
    print("Classification in progress, please wait...")
    TRAINING_SET_SIZE = 90
    TEST_IMAGES = [95, 195, 96, 97, 198, 88, 98, 99, 94, 191]

    # get training data in the form of feature descriptors
    # training_set, training_labels = Data.create_hog_data_set(TRAINING_SET_SIZE)
    # test_images = Data.create_hog_image(TEST_IMAGES)
    training_set, training_labels = Data.create_realistic_hog_data_set(TRAINING_SET_SIZE)
    test_images = Data.create_realistic_hog_test_set(TEST_IMAGES)

    clf = svm.SVC(gamma=0.0001, C=10)

    # fit training data
    clf.fit(training_set, training_labels)

    result = clf.predict(test_images)
    print("Prediction: ", result)

    # GUI
    MAX_COLUMNS = 5
    MAX_ROWS = 2

    for i in range(1, len(test_images) + 1):
        plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
        plt.imshow(Parser.load_full_size_image(TEST_IMAGES[i - 1]), cmap="gray")
        if result[i - 1] == 1:
            plt.title("Aircraft")
        else:
            plt.title("Ground")
        plt.axis('off')

    plt.show()

elif CLASSIFICATION_METHOD == 2:
    while True:
        IMAGE_SIZE = input("Enter image search dimension(pixels): ")
        if IMAGE_SIZE.isdigit():
            IMAGE_SIZE = int(IMAGE_SIZE)
            break

    while True:
        X_STEP = input("Enter search step X (pixels): ")
        Y_STEP = input("Enter search step Y (pixels): ")
        if X_STEP.isdigit() and Y_STEP.isdigit():
            X_STEP = int(X_STEP)
            Y_STEP = int(Y_STEP)
            break

    print("search in progress....")

    training_set, training_labels = Data.create_resized_hog_data_set(IMAGE_SIZE)
    test_images = Data.create_airport_hog_data_set(2, X_STEP, Y_STEP, IMAGE_SIZE)

    clf = svm.SVC(gamma=0.0001, C=10)

    # fit training data
    clf.fit(training_set, training_labels)

    result = clf.predict(test_images)

    for i in result:
        print(i)
    print("Prediction: ", result)
