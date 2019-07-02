import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

import os
from glob import glob

import tensorflow.keras.layers as klayer

#from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization
from keras.models import Model
#from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2

from keras_contrib.losses.jaccard import jaccard_distance as jaccard
from keras import metrics

from keras.callbacks import Callback
import keras.backend as K

from tensorboard.plugins.beholder import Beholder

import json
with open(os.path.join(os.getcwd(), "config.json"), mode="r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
mask_csv_filename = config["mask_csv_filename"]


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


class BeholderCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.beholder = Beholder(log_dir)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None, *args, **kwargs):
        pass

    def on_train_batch_end(self, batch, logs=None, *args, **kwargs):
        K = keras.backend.backend
        sess = keras.backend.get_session()
        self.beholder.update(session=sess)


def mod_jaccard(y_true, y_pred, smooth=1):
    K = keras.backend
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (jac) * smooth


def unet(learning_rate, pretrained_weights=None, input_size=(256, 256, 1), down_sampling=4, give_intermediate=False, main_activation="relu", k_initializer="he_normal", zero_weight = 0.5):
    """Directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    conv1 = klayer.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = klayer.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = klayer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = klayer.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = klayer.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = klayer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = klayer.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = klayer.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = klayer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = klayer.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = klayer.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = klayer.Dropout(0.5)(conv4)
    pool4 = klayer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = klayer.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = klayer.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = klayer.Dropout(0.5)(conv5)

    up6 = klayer.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(klayer.UpSampling2D(size = (2,2))(drop5))
    merge6 = klayer.concatenate([drop4,up6], axis = 3)
    conv6 = klayer.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = klayer.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = klayer.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(klayer.UpSampling2D(size = (2,2))(conv6))
    merge7 = klayer.concatenate([conv3,up7], axis = 3)
    conv7 = klayer.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = klayer.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = klayer.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(klayer.UpSampling2D(size = (2,2))(conv7))
    merge8 = klayer.concatenate([conv2,up8], axis = 3)
    conv8 = klayer.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = klayer.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = klayer.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(klayer.UpSampling2D(size = (2,2))(conv8))
    merge9 = klayer.concatenate([conv1,up9], axis = 3)
    conv9 = klayer.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = klayer.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = klayer.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = klayer.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    out_layer = conv10

    layers = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, out_layer]

    model = tf.keras.Model(inputs=inputs, outputs=layers if give_intermediate else out_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.MSE, metrics=[metrics.binary_accuracy, mod_jaccard])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_orig(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4, give_intermediate=False, main_activation="relu", k_initializer="he_normal", zero_weight = 0.5):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv1 = klayer.Conv2D(64, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(scaled_inputs)
    conv1 = klayer.Conv2D(64, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv1)
    pool1 = klayer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = klayer.Conv2D(128, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(pool1)
    conv2 = klayer.Conv2D(128, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv2)
    pool2 = klayer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = klayer.Conv2D(256, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(pool2)
    conv3 = klayer.Conv2D(256, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv3)
    pool3 = klayer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = klayer.Conv2D(512, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(pool3)
    conv4 = klayer.Conv2D(512, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv4)
    drop4 = klayer.Dropout(0.5)(conv4)
    pool4 = klayer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = klayer.Conv2D(1024, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(pool4)
    conv5 = klayer.Conv2D(1024, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv5)
    drop5 = klayer.Dropout(0.5)(conv5)

    up6 = klayer.Conv2D(512, 2, activation=main_activation, padding='same', kernel_initializer=k_initializer)(
        klayer.UpSampling2D(size=(2, 2))(drop5))
    merge6 = klayer.concatenate([drop4, up6], axis=3)
    conv6 = klayer.Conv2D(512, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(merge6)
    conv6 = klayer.Conv2D(512, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv6)

    up7 = klayer.Conv2D(256, 2, activation=main_activation, padding='same', kernel_initializer=k_initializer)(
        klayer.UpSampling2D(size=(2, 2))(conv6))
    merge7 = klayer.concatenate([conv3, up7], axis=3)
    conv7 = klayer.Conv2D(256, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(merge7)
    conv7 = klayer.Conv2D(256, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv7)

    up8 = klayer.Conv2D(128, 2, activation=main_activation, padding='same', kernel_initializer=k_initializer)(
        klayer.UpSampling2D(size=(2, 2))(conv7))
    merge8 = klayer.concatenate([conv2, up8], axis=3)
    conv8 = klayer.Conv2D(128, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(merge8)
    conv8 = klayer.Conv2D(128, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv8)

    up9 = klayer.Conv2D(64, 2, activation=main_activation, padding='same', kernel_initializer=k_initializer)(
        klayer.UpSampling2D(size=(2, 2))(conv8))
    merge9 = klayer.concatenate([conv1, up9], axis=3)
    conv9 = klayer.Conv2D(64, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(merge9)
    conv9 = klayer.Conv2D(32, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv9)
    conv9 = klayer.Conv2D(16, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(conv9)
    conv10 = klayer.Conv2D(1, 1, activation=None)(conv9)

    out_layer = klayer.UpSampling2D(size=(down_sampling, down_sampling))(conv10)

    layers = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, out_layer]

    model = tf.keras.Model(inputs=inputs, outputs=layers if give_intermediate else out_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=create_weighted_binary_crossentropy(zero_weight=zero_weight, one_weight=1-zero_weight), metrics=["accuracy"])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def smet(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4, give_intermediate=False, main_activation="sigmoid", k_initializer="he_normal", zero_weight = 0.5):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv3 = klayer.Conv2D(1, 1, activation="relu", use_bias=True, padding='same', kernel_initializer=k_initializer)(inputs)
    #conv3 = klayer.Conv2D(1, 1, activation=main_activation, use_bias=True, padding='same', kernel_initializer=k_initializer)(conv3)
    out_layer = klayer.UpSampling2D(size=(down_sampling, down_sampling))(conv3)
    layers = [conv3, out_layer]

    model = tf.keras.Model(inputs=inputs, outputs=layers if give_intermediate else conv3)

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.MSE, metrics=[metrics.binary_accuracy, mod_jaccard])

    # model.summary()
    #all_layers_output = keras.backend.function([model.layers[0].input],
    #           [l.output for l in model.layers[1:]])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model#, all_layers_output


def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:
        Model        : A Keras model instance
    """

    if nb_classes == None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

    if compression <= 0.0 or compression > 1.0:
        raise Exception(
            'Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')

    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]

    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2

    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')

    # Initial convolution layer
    x = Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1),
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    # Building dense blocks
    for block in range(dense_blocks):

        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck,
                                     weight_decay)

        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'

    if bottleneck:
        model_name = model_name + 'b'

    if compression < 1.0:
        model_name = model_name + 'c'

    return Model(img_input, x, name=model_name), model_name


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """

    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


class DataLoader(keras.utils.Sequence):
    def __init__(self, filepaths, batch_size=32, dim=(1024, 1024), shuffle=True, mask_dir=""):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.filepaths = filepaths
        self.mask_dir = mask_dir
        self.indices = np.arange((len(self.filepaths)))
        self.shuffle = shuffle
        self.on_epoch_end()
        self.csv = pd.read_csv(mask_csv_filename, header=None, index_col=0)
        with open(mask_csv_filename, mode="r") as f:
            raw_rle = f.read()

        self.rles = {}

        for line in raw_rle.split("\n"):
            key, rle_data = line.split(",")
            rle_data = [x for x in rle_data.split() if x != ""]
            self.rles[key] = rle_data

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, filepaths):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1), dtype=np.uint8)
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=np.bool)

        # Generate data
        for i, filepath in enumerate(filepaths):
            # Store sample
            X[i,], Y[i] = self.load_filepath(filepath)

        return X, Y

    def load_filepath(self, filepath):
        a = np.asarray(Image.open(filepath[:-4]+".png"))
        a = preprocess_image(a)
        X = a[::4,::4]
        X = (np.expand_dims(X, axis=2).astype(np.float32) + 1) / 2
        
        Y = np.load(os.path.join(self.mask_dir, filepath.split(os.sep)[-1])[:-4])#.T
        #y_rle = self.rles[filepath.split(os.sep)[-1][:-4]]
        #Y = rle2mask(y_rle, *self.dim).T
        #  Y = np.reshape(Y, self.dim)
        #  Y = np.expand_dims(Y, axis=2).astype(np.float)
        #  Y = (X>0.5).astype(np.float)
        return X, X.copy()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_shuffled_files = [self.filepaths[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(batch_shuffled_files)

        return X, Y


def check_valid_datafile(filepath, rle_df, needs_label=True):
    # We try to load each file in order to find which ones are invalid
    try:
        xray_image = Image.open(filepath)
    except Exception as e:
        print(e)
        print(f"Skipping loading of {filepath}, file didn't load correctly")
        return False

    if needs_label:
        try:
            str(rle_df.loc[filepath.split(os.sep)[-1][:-4], 1])
        except Exception as e:
            print(e)
            print(f"Skipping loading of {filepath}, file doesn't have label when it should")
            return False
    return True


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    if rle[0] == "-1":
        return mask
    split_data = [x for x in rle if x.isdigit()]
    array_data = [int(x) for x in split_data]
    array = np.asarray(array_data)
    starts = array[0::2]
    lengths = array[1::2]
    if len(starts) > len(lengths):
        starts = starts[:len(lengths)]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:min(current_position + lengths[index], (width * height) - 1)] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# def show_dcm_info(dataset):
#     print("Filename.........:", file_path)
#     print("Storage type.....:", dataset.SOPClassUID)
#     print()
#
#     pat_name = dataset.PatientName
#     display_name = pat_name.family_name + ", " + pat_name.given_name
#     print("Patient's name......:", display_name)
#     print("Patient id..........:", dataset.PatientID)
#     print("Patient's Age.......:", dataset.PatientAge)
#     print("Patient's Sex.......:", dataset.PatientSex)
#     print("Modality............:", dataset.Modality)
#     print("Body Part Examined..:", dataset.BodyPartExamined)
#     print("View Position.......:", dataset.ViewPosition)
#
#     if 'PixelData' in dataset:
#         rows = int(dataset.Rows)
#         cols = int(dataset.Columns)
#         print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
#             rows=rows, cols=cols, size=len(dataset.PixelData)))
#         if 'PixelSpacing' in dataset:
#             print("Pixel spacing....:", dataset.PixelSpacing)

def preprocess_image(image_array: np.ndarray):
    scaled_image_array = (image_array.astype(np.float16) - 128) / 128
    return scaled_image_array


def preprocess_mask(mask: np.ndarray):
    #mask = mask-128
    #mask = mask/128
    return mask


if __name__ == "__main__":
    keras.backend.clear_session()
    train_data_pref = data_dir + "train_png"
    test_data_pref = data_dir + "test_png"

    try:
        with open(data_dir + "valid_train_filepaths", mode="r") as f:
            valid_train_filepaths = f.read().split("\n")
        print("Loaded data from old list of valid data files")
    except FileNotFoundError:
        print("Re-checking which data files are valid")
        print(os.path.join(mask_csv_filename))
        rle_data = pd.read_csv(mask_csv_filename, header=None, index_col=0)
        valid_train_filepaths = [file_path[:-4]+".npy" for file_path in
                                 glob(os.path.join(train_data_pref, "*.png"), recursive=True)
                                 if check_valid_datafile(file_path, rle_data)]
        valid_test_filepaths = [file_path[:-4]+".npy" for file_path in
                                glob(os.path.join(test_data_pref, "*.png"), recursive=True)
                                if check_valid_datafile(file_path, rle_data, needs_label=False)]

        print(len(valid_test_filepaths), len(valid_train_filepaths))

        print("Converting all files into numpy-native format")

        def store_np_file(filepath):
            pass#a = np.asarray(Image.open(filepath[:-4]+".png"))
            #a = preprocess_image(a)
            #a = np.expand_dims(a, axis=2)
            #np.save(filepath, a)

        import sys

        for inx, f in enumerate(valid_test_filepaths):
            store_np_file(f)
            if inx % 100 == 0:
                print(inx)
        for inx, f in enumerate(valid_train_filepaths):
            store_np_file(f)
            if inx % 100 == 0:
                print(inx)

        with open(data_dir + "valid_train_filepaths", mode="w") as f:
            f.write("\n".join(valid_train_filepaths))
        with open(data_dir + "valid_test_filepaths", mode="w") as f:
            f.write("\n".join(valid_test_filepaths))

    with open(mask_csv_filename, mode="r") as f:
        raw_rle = f.read()

    def check_store_mask_file(raw_csv_data):
        key, rle_d = raw_csv_data.split(",")
        try:
            with open(os.path.join(data_dir, "train_png", "masks", key), mode="r") as f:
                pass
        except FileNotFoundError:
            rle_d = [x for x in rle_d.split() if x != ""]
            mask = rle2mask(rle_d, 1024, 1024).astype(np.bool)
            mask = preprocess_mask(mask)
            mask = mask.reshape((1024, 1024)).T
            #plt.imshow(mask)
            #plt.show()
            with open(os.path.join(data_dir, "train_png", "masks", key), mode="wb") as f:
                np.save(f, mask)

        return os.path.join(data_dir, "train_png", "masks", key)

    mask_processing_pool = Pool(cpu_count())
    try:
        with open(os.path.join(data_dir, "valid_mask_paths"), mode="r") as f:
            valid_mask_paths = f.read().split("\n")
            print("Used precomputed valid mask paths")
    except FileNotFoundError:
        valid_mask_paths = []
        print("Computing valid mask paths")
        for inx, key in enumerate(raw_rle.split("\n")):
            valid_mask_paths.append(check_store_mask_file(key))
            if inx % 100 == 0:
                print(inx)

        with open(os.path.join(data_dir, "valid_mask_paths"), mode="w") as f:
            f.write("\n".join(valid_mask_paths))

        print("Done computing valid mask paths")

    print("Setup done!")

    # Network and training params/config
    dims = (256,256)
    n_epochs = 10
    batch_size = 1
    img_downsampling = 16
    learning_rate = 1e-4
    num_train_examples = 10
    use_validation = False
    validation_coeff = 0.1
    retrain = True
    net_arch = "unet"

    # The file in which trained weights are going to be stored
    net_filename = f"{net_arch}-epochs_{n_epochs}-batchsz_{batch_size}-lr_{learning_rate}-downsampling_{img_downsampling}-numexamples_{num_train_examples}"


    with open(mask_csv_filename, mode="r") as f:
            raw_rle = f.read()
    raw_csv_data = raw_rle.split("\n")
    haspneumo_lookup = {}
    for line in raw_csv_data:
        key,rle = line.split(",")
        haspneumo_lookup[key] = False if rle.strip() == "-1" else True
    
    pneumo_filepaths = []
    for path in valid_train_filepaths:
        if haspneumo_lookup[os.path.split(path)[1][:-4]]:
            pneumo_filepaths.append(path)

    use_filepaths = pneumo_filepaths
    num_validation_examples = int(num_train_examples * validation_coeff)
    train_filepaths = use_filepaths[:-int(validation_coeff*len(use_filepaths)) if use_validation else len(use_filepaths)][:num_train_examples]
    validation_filepaths = use_filepaths[-int(validation_coeff*len(use_filepaths)):][:num_validation_examples]
    model = locals()[net_arch](down_sampling=img_downsampling, learning_rate=learning_rate)

    try:
        if retrain:
            raise Exception
        print("Loading pretrained network weights, from", os.path.join(data_dir + "models", net_filename))
        model.load_weights(os.path.join(data_dir + "models", net_filename))
    except Exception as e:
        print("Was not able to load model...")
        print("Training network!")

        train_generator = DataLoader(train_filepaths, dim=(256, 256), batch_size=batch_size,
                                     mask_dir=os.path.join(data_dir, "train_png", "masks"))
        if use_validation:
            validation_generator = DataLoader(validation_filepaths, batch_size=batch_size,
                                          mask_dir=os.path.join(data_dir, "train_png", "masks"))

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=config["logdir"], histogram_freq=1, write_grads=True,
                                                      write_graph=True, write_images=True),
            BeholderCallback(log_dir=config["logdir"]),
            keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4)
        ]

        print("Fitting")
        try:
            please_stop = False
            model.fit_generator(train_generator, validation_data=validation_generator if use_validation else None,
                                validation_freq=1 if use_validation else None,
                                epochs=n_epochs, use_multiprocessing=True, callbacks=callbacks)
        except KeyboardInterrupt:
            please_stop = True
        except Exception as e:
            raise e
        finally:
            print("Saving model weights in", os.path.join(data_dir, "models", net_filename))
            model.save_weights(os.path.join(data_dir, "models", net_filename))
            print("Done saving model!")
            if please_stop:
                exit()
    train_generator = DataLoader(train_filepaths, batch_size=batch_size, mask_dir=os.path.join(data_dir, "train_png", "masks"))
    prediction_model = locals()[net_arch] \
        (down_sampling=img_downsampling, learning_rate=learning_rate, give_intermediate=False)

    del model

    print("Plotting predictions...")

    # Visualisation config
    images_per_row = 16
    layers_to_vis = []#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    figure_suffix = ".png"

    import math
    def plot_hidden_layer_activations(pred, layer_inx, imgs_per_row=None, filename=""):
        if not imgs_per_row:
            imgs_per_row = images_per_row
        # Displays the feature maps
        n_features = pred.shape[-1]  # Number of features in the feature map
        size = pred.shape[0]  # The feature map has shape (1, size, size, n_features).
        n_cols = math.ceil(n_features / imgs_per_row) # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, imgs_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(imgs_per_row):
                try:
                    channel_image = pred[:, :, col * imgs_per_row + row]
                except IndexError:
                    break
                # channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                # channel_image /= channel_image.std()
                # channel_image *= 64
                # channel_image += 128
                # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.grid(False)
        plt.title("Layer nr. " + str(layer_inx + 1))
        plt.imshow(display_grid, aspect='auto', cmap='bone')
        plt.savefig(os.path.join(data_dir, "plots", "Layer_" + str(layer_inx+1) + "_" + filename + figure_suffix), bbox_inches="tight")

    for to_predict in train_filepaths:
        print("Plotting predictions/layers from", os.path.split(to_predict)[1], "...")
        x, y = train_generator.load_filepath(to_predict)

        pred_layers = prediction_model.predict(np.asarray([x]))
        plt.grid(False)
        plt.title("Input")
        plt.imshow(x.squeeze().astype(np.float32), aspect='auto', cmap='bone')
        plt.savefig(os.path.join(data_dir, "plots", "Input_" + os.path.split(to_predict)[1] + figure_suffix), bbox_inches="tight")
        plt.close('all')
        for layer_inx in layers_to_vis:
            print("\tOn layer nr.", layer_inx+1)
            pred = pred_layers[layer_inx][0]
            plot_hidden_layer_activations(pred, layer_inx, filename=os.path.split(to_predict)[1])
            plt.close('all')
        
        plt.subplot(2, 2, 1)
        plt.imshow(pred_layers.squeeze().round(), vmin=0, vmax=1)
        plt.subplot(2, 2, 2)
        plt.imshow(x.squeeze().astype(np.float32), vmin=0, vmax=1)
        plt.subplot(2, 2, 3)
        plt.imshow(pred_layers.squeeze(), vmin=0, vmax=1)
        plt.subplot(2, 2, 4)
        plt.imshow(y.squeeze(), vmin=0, vmax=1)
        plt.show()

