from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from data_parser import Data

training_set, training_labels = Data.new_full_size_training_set(90)
samples = len(training_set)
training_set = training_set.reshape((samples, -1))

X_train, X_test, y_train, y_test = \
    train_test_split(training_set, training_labels, test_size=0.5, random_state=0)

tuning_parameters = [
    {'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e-10],
     'C': [0.001, 0.1, 1, 10, 100, 1000]}]

classifier = GridSearchCV(SVC(), tuning_parameters, scoring="precision", cv=5)

classifier.fit(X_train, y_train)

print(classifier.best_params_)
