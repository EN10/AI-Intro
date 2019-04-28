# Simple AI Intro

Some of the key Libraries and Dataset used:     
* [Tensorflow](https://en.wikipedia.org/wiki/TensorFlow) as an AI Library created by Google and can be used in Python.    
* [Keras](https://en.wikipedia.org/wiki/Keras) Also a Python AI library, it is used on top of Tensorflow. It allows models to be created faster and more easily than with Tensorflow alone.   
* [MNIST](https://en.wikipedia.org/wiki/MNIST_database) A collection of 70,000 handwritten digits from 0-9, it is a commonly used dataset for training neural networks.   
![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "MNIST")    


```python
# Point to MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist", one_hot=True)

import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(10, activation='softmax', input_shape=(784,)))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(mnist.train.images, mnist.train.labels,
          epochs=5)

model.evaluate(mnist.test.images, mnist.test.labels)
```

## References

* [EN10 Keras MNIST](https://github.com/EN10/KerasMNIST)
* [Keras Docs](https://keras.io/getting-started/sequential-model-guide)
* [Tensorflow Fashion MNIST](https://www.tensorflow.org/tutorials/keras/basic_classification)
* [Udacity TF Intro](https://eu.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
