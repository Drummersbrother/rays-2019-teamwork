import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

mnist = keras.datasets.mnist
(trainingimages, traininglabels), (testimages, testlabels) = mnist.load_data()

encodedlabels = tf.keras.utils.to_categorical(traininglabels)


classnames = ["0","1","2","3","4","5","6","7","8","9"]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(100, activation=tf.nn.sigmoid), 
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(trainingimages, encodedlabels, epochs = 5)

