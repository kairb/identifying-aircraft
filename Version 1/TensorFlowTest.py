# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
import random

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST training set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ensuring training labels are same length as images array
# print(train_images.shape, test_images.shape)
# print(len(train_labels), len(test_labels))

# Training images are preprocessed
train_images = train_images / 255.0

test_images = test_images / 255.0

# First 25 images of dataset are displayed with appropriate lables
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# contstruct model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train model
model.fit(train_images, train_labels, epochs=1)

# test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

