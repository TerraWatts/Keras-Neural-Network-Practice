'''

Practice Exercise: Neural Networks with Keras
Watts Dietrich
Oct 23 2020

The goal of this exercise is to build a neural network to classify images of clothing from the fashion MNIST dataset.
This dataset contains 70,000 28x28-pixel images of clothing items, each belonging to one of ten categories.
60,000 images are used for training, 10,000 for testing.

The network uses:
784 input nodes (one for each pixel of the images)
A single 128-node hidden layer
10 output nodes (one for each possible category of clothing)

After training, an accuracy of roughly 88% is obtained.
The program shows a few images along with their predicted and actual clothing category for user human validation.

'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# get dataset
data = keras.datasets.fashion_mnist

# split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# these names are associated with the 10 numerical category labels in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# this will print the raw data of one of the training images. it is a 28x28 array of pixel color values, 0-255
#print(train_images[7])

# here we convert the 0-255 pixel color values to 0-1 range
train_images = train_images/255.0
test_images = test_images/255.0

# show one of the training images
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

# Build the model
# The 28x28 images will be flattened into a 1d list of 784 elements and fed into 784 input nodes
# 10 output nodes are used because there are 10 categories of clothing. Model will assign probabilities to each.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Input layer, flattens 28x28 pixel image to 784 input nodes
    keras.layers.Dense(128, activation="relu"), # Hidden layer
    keras.layers.Dense(10, activation="softmax") # Output layer. Softmax ensures output values will sum to 1.
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# train the model
model.fit(train_images, train_labels, epochs=10)

# test model accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc:", test_acc)

# get model predictions
prediction = model.predict(test_images)

# for human validation, show a few images along with their predicted and actual clothing categories
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()