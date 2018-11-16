#import statements
from sklearn import datasets
from sklearn import svm

import matplotlib.pyplot as plt


digits = datasets.load_digits()


clf = svm.SVC(gamma="auto", C=100)

x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)

#print("Prediction: ", clf.predict(digits.data[-1]))

plt.imshow(digits.images[-1], cmap = plt.cm.gray_r, interpolation = "nearest")

plt.show()
