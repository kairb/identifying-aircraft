#import statements
from sklearn import datasets
from sklearn import svm

import matplotlib.pyplot as plt
from image_parser import Parser
import numpy as np

clf = svm.SVC()

image = Parser.load_image(1)

#print(image[1])

digits = datasets.load_digits()

x,y = digits.data[:-1], digits.target[:-1]
print(x[1])

array = np.concatenate(Parser.load_image(1),Parser.load_image(2))
# for i in range(1,100):


# clf.fit(Parser.load_image(i),1)



# print("Prediction: ",clf.predict(Parser.load_image(100)))


#plt.imshow(Parser.load_image(100), cmap = plt.cm.gray_r, interpolation = "nearest")

#plt.show()


