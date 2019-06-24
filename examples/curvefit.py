#!/usr/bin/python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    return 0.8*np.tanh(2*x+0.5)+0.1

# Data
num_points = 100
#x = np.concatenate((np.linspace(0.01, 0.5, num_points), np.linspace(0.5, 1, 5)))
x = np.linspace(0.0, 1, num_points)
y = fun(x)
x_plot = np.linspace(0.0, 1, num_points)
y_plot = fun(x_plot)

#plt.plot(x, y, '.-')
#plt.show()

# Neural network model
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(1,), activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.tanh),
    keras.layers.Dense(1),
])

# Parameters for optimization
model.compile(optimizer='adam',#keras.optimizers.SGD(lr=0.10),
              loss='mse')

# Train model
model.fit(x, y, epochs=2000)

# Check model on other data
dx = x_plot[1] - x_plot[0]
x_test = x_plot + 0.5*dx
y_test = fun(x_test)
loss = model.evaluate(x_test, y_test)
print('Test RMS error:', np.sqrt(loss))

# Plot function values
N = model.predict(x)
N_test = model.predict(x_test)
#print(model.get_weights())

plt.plot(x_plot, y_plot)
plt.scatter(x, N)
plt.plot(x_test, N_test)
plt.show()
