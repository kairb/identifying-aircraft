from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from data_parser import Data
from sklearn.model_selection import cross_val_score

sizes = [300, 150, 75]
for size in sizes:

    training_set, training_labels = Data.create_training_data(size, size)

    classifier = SVC(C=10, gamma=0.0001)
    scores = cross_val_score(classifier, training_set, training_labels, cv=10)
    print(size, "x", size)

    total = 0
    for score in scores:
        total += score

    print(scores)
    print("Average: ", total / len(scores) , "\n")
