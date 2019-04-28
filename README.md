# Simple AI Intro

```python
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
