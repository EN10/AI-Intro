# AI Intro

Some of the key Libraries and Dataset used:     
* [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) a free progarmming environment. 
* [Tensorflow](https://en.wikipedia.org/wiki/TensorFlow) as an AI Library created by Google and can be used in Python.    
* [Keras](https://en.wikipedia.org/wiki/Keras) Also a Python AI library, it is used on top of Tensorflow. It allows models to be created faster and more easily than with Tensorflow alone.   
* [MNIST](https://en.wikipedia.org/wiki/MNIST_database) A collection of 70,000 handwritten digits from 0-9, it is a commonly used dataset for training neural networks.   
![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "MNIST")    

## Code Example
```python
# Point to the MNIST tensorflow directory
from tensorflow.examples.tutorials.mnist import input_data
# Downloads and formats MNIST
mnist = input_data.read_data_sets("./mnist", one_hot=True)

import keras
model = keras.models.Sequential()
# See model Image below, 784 pixels on the left and 10 neurons on the right
model.add(keras.layers.Dense(10, activation='softmax', input_shape=(784,)))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains the model by feeding 60,000 images
model.fit(mnist.train.images, mnist.train.labels,
          epochs=5)

# Tests the model with 10,000 new test images
model.evaluate(mnist.test.images, mnist.test.labels)
```
Model: ![alt text](https://ml4a.github.io/images/figures/mnist_1layer.png "Model")  
[Softmax](https://en.wikipedia.org/wiki/Softmax_function)   
[Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)  

## References

* [EN10 Keras MNIST](https://github.com/EN10/KerasMNIST)
* [Keras Docs](https://keras.io/getting-started/sequential-model-guide)
* [Tensorflow Fashion MNIST](https://www.tensorflow.org/tutorials/keras/basic_classification)
* [Udacity TF Intro](https://eu.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
