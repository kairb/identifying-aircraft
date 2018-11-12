# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
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

# construct model
image_shape = (400, 400, 3)
model_1 = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                                      input_shape=image_shape, pooling= None)

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
model.fit(train_images, train_labels, epochs=3)


print("test")
# test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

