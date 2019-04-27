# Simple-AI

```python {.line-numbers}
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
