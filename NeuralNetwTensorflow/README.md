# AI 

## Tensorflow / Keras

`TensorFlow` is an open-sourced end-to-end platform, a library for multiple machine learning tasks, while `Keras` is a high-level neural network library that runs on top of TensorFlow. Both provide high-level APIs used for easily building and training models, but Keras is more user-friendly because it's built-in Python.
https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article

## Prerequisites

pip install :
1. numpy
2. tensorflow
3. matplotlib
   
Or get them all from Anaconda (Anaconda | The World's Most Popular Data Science Platform): https://www.anaconda.com/

Check if you have those libraries installed:
```
pip list | grep numpy
numpy          1.20.1
```

## Programming Flow

1. import all of our necessary downloaded libraries
2. get the training and testing dataset, fashion_mnist (which contains 60k 28x28 images): https://keras.io/api/datasets/fashion_mnist/
3. we can take a look at what the data looks like (display)... sampling the data
4. Let's create a neural net for our analysis
   1. Flatten the input from a 28x28 image into 784x1 input layer for the neural net
   2. Output is 10 labels (0-9), identifying a t-shirt, trouser, pullover, dress, etc. (found on keras link). Output layer will contain 10 nodes
   3. Hidden layer will contain 128 nodes. Dense is used to connect all nodes from the previous layer to all the nodes in the next layer
5. Every node connection/edge contains a weight and bias, which is multiplied and added (respectively) to the node value. Throughout the numerous amounts of weights and bias, the output layer will basically be the result of a huge and complex "function" that captures many patterns throughout the image
6. Compile neural network to be ready to be trained
   1. Optimizer will tweak the neural network to have a lower loss/error function. Basically, the optimizer optimizes the neural network by changing its weights and biases. If you guys ever want to know more about how an optimizer works, basically, it can take an errorful prediction, and lower the weights that associate with that value, and amplify the weights that correspond with the correct value. Through a whole process of minimization of the loss function, the optimizer can find the best neural network weights and biases for our application.
   2. The loss function we are using is a function made to identify errors in predictions by our neural net. Put simply, it tells us how wrong our neural network is. Sadly, I don't know how this tensorflow loss function exactly works, but we don't have to.
7. Train our data to our data and labels. Epochs are the amount of times we reconstruct our optimized model based on its loss.
8. Test our data on our final model after training. Let's see how wrong we are after all of our epochs of training
9. Let's visualize how accurate how model is on the test data by printing the first test label expected prediction and use our model to make a prediction about the first test label