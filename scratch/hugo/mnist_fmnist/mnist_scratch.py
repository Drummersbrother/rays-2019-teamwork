import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from tensorflow import keras

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 3} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

def show_pic(pic):
    plt.figure()
    plt.imshow(pic)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_pics(pics, labels):
    plt.figure(figsize=(np.ceil(np.sqrt(len(pics))), np.ceil(np.sqrt(len(pics)))))
    for inx, val in enumerate(pics):
        plt.subplot(*(np.ceil(np.sqrt(len(pics))), np.ceil(np.sqrt(len(pics)))), inx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(val)
        plt.xlabel(str(labels[inx]))
    plt.show()


def get_onehot_encoded(indices):
    onehot = np.zeros((indices.size, indices.max() + 1))
    onehot[np.arange(indices.size), indices] = 1
    return onehot


def get_test_acc(model, x, y):
    preds = model.predict(x)
    n_correct = sum([a == b for a, b in zip(np.argmax(preds, 1), y)]).sum()
    return n_correct / len(x)


def get_model(use_saved=False, model_fname=None):
    if use_saved:
        try:
            return keras.models.load_model("models/" + model_fname)
        except OSError:
            pass
    # Our network has a flattening layer, as we don't keep spatial data
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10), keras.layers.LeakyReLU(),
        keras.layers.Dense(10), keras.layers.LeakyReLU(),
        keras.layers.Dense(10), keras.layers.LeakyReLU(),
        keras.layers.Dense(10), keras.layers.Softmax()
    ])

    return model


if __name__ == "__main__":

    lr = 1
    n_epochs = 40

    # Following the tutorial on https://www.tensorflow.org/tutorials/keras/basic_classification
    mnist = keras.datasets.mnist
    (train_images, train_labels_inxs), (test_images, test_labels_inxs) = mnist.load_data()

    # We rescale the data from 0-255 to 0-1
    train_images, test_images = train_images / 255, test_images / 255

    # We edit the label data as to be one-hot
    train_labels, test_labels = get_onehot_encoded(train_labels_inxs), get_onehot_encoded(test_labels_inxs)

    # Our network has a flattening layer, as we don't keep spatial data
    model_filename = "simple_mnist_dense" + f"_{n_epochs}e_{lr}lr"
    model = get_model(True, model_filename)

    # We choose the "functional" hyperparameter
    #with tf.device("/cpu:0"):
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss="mse")

    # We only train the network on training data
    #with tf.device("/gpu:0"):
    model_history = model.fit(train_images, train_labels, epochs=n_epochs, use_multiprocessing=True)

    model.save("models/"+model_filename)

    # We let the network do predictions on the whole validation set and measure basically the top1error
    print(f"Test accuracy: {round(100*get_test_acc(model, test_images, test_labels_inxs), 2)}")

    # Plot training & validation loss values
    plt.plot(model_history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss (MSE)')
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
