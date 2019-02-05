from sklearn import svm
from test_set import create_test_set, create_white_image, create_black_image
import numpy as np

images, labels = create_test_set(200)
clf = svm.SVC(C=1, gamma='auto')

clf.fit(images, labels)
test = []
test.append(create_black_image())
test.append(create_white_image())
test.append(create_black_image())

print(clf.predict(test))
