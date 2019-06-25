import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.cm as mplcm
from matplotlib import pyplot as plt
from tensorflow import keras


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
        plt.imshow(val, cmap=mplcm.gray)
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
        keras.layers.Dense(728), keras.layers.LeakyReLU(),
        keras.layers.Dense(100), keras.layers.LeakyReLU(),
        keras.layers.Dense(25), keras.layers.LeakyReLU(),
        keras.layers.Dense(10), keras.layers.Softmax()
    ])

    return model


if __name__ == "__main__":

    lr = 0.001
    n_epochs = 200

    # Following the tutorial on https://www.tensorflow.org/tutorials/keras/basic_classification
    mnist = keras.datasets.fashion_mnist
    (train_images, train_labels_inxs), (test_images, test_labels_inxs) = mnist.load_data()
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

    # We rescale the data from 0-255 to 0-1
    train_images, test_images = train_images / 255, test_images / 255

    # We edit the label data as to be one-hot
    train_labels, test_labels = get_onehot_encoded(train_labels_inxs), get_onehot_encoded(test_labels_inxs)

    # Our network has a flattening layer, as we don't keep spatial data
    model_filename = "simple_fashion_mnist_dense" + f"_{n_epochs}e_{lr}lr"
    model = get_model(True, model_filename)

    # We choose the "functional" hyperparameters
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss="mse")

    # We only train the network on training data
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

    # We find some examples that are misclassified by the network and display them
    preds = model.predict(test_images)

    failed_indices = []
    for inx, (a, b) in enumerate(zip(np.argmax(preds, 1), np.argmax(test_labels, 1))):

        if not a == b:
            failed_indices.append(inx)
    failed_indices = failed_indices[:25]

    show_pics([test_images[a] for a in failed_indices],
              [class_names[int(np.argmax(test_labels[a]))] + "-" + class_names[int(np.argmax(preds[a]))] for a in failed_indices])
