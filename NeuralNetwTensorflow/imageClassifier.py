import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

#labeled dataset (70k images, 60k training, 10k testing)
fashion_mnist = keras.datasets.fashion_mnist

#load our fashion data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# our fashion data visualization print
print(train_labels[0])
print(train_images[0])
plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

#neural network model construction
model = keras.Sequential([
    #input layer: 28x28 image array flattened into 1d vector with 784x1 height
    keras.layers.Flatten(input_shape = (28,28)),

    #hidden layer: 128 nodes, relu returns value of the node, or 0 if the value is negative
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    #output layer: 0-9, dense connects all of the nodes from the previous layer to this layer created, returns the maximum node probability of the 10 nodes
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# compile our keras build neural net model to make it ready to be trained
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train!
model.fit(train_images, train_labels, epochs=5)

# test!
test_loss = model.evaluate(test_images, test_labels)




# visualization of trained neural network on the test data
plt.imshow(test_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()
print("Expected Classification:", test_labels[0])

#predict!
predictions = model.predict(test_images)
#visualize our predictions to the console!
print("Our prediction probabilties for the first image:", predictions[0])

#this is our actual label prediction based on the neural network, it is the index of max of the predictions[0]
print("Our actual prediction:", list(predictions[0]).index(max(predictions[0])))