"""Imports"""
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
import pandas as pd
from matplotlib import pyplot as plt

import os
from glob import glob

import keras.layers as klayer
from keras.applications.xception import Xception

# from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization
from keras.models import Model
# from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2

from keras_contrib.losses.jaccard import jaccard_distance as jaccard
from keras import metrics

from keras.callbacks import Callback
import keras.backend as K

"""Define losses and metrics"""


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return keras.backend.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return keras.backend.binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


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


def mod_jaccard(y_true, y_pred, smooth=1):
    K = keras.backend
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (jac) * smooth



"""Define neural network models"""


def unet(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4, give_intermediate=False,
         main_activation="relu", k_initializer="he_normal", zero_weight=0.5):
    """Directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    actual_down_sampling = 2
    actual_pooling = (actual_down_sampling, actual_down_sampling)
    inputs = klayer.Input(input_size)
    scaled_ins = klayer.MaxPooling2D(pool_size=actual_pooling)(inputs)
    conv1 = klayer.Conv2D(64, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(scaled_ins)
    conv1 = klayer.Conv2D(64, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = klayer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = klayer.Conv2D(128, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = klayer.Conv2D(128, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = klayer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = klayer.Conv2D(256, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = klayer.Conv2D(256, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = klayer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = klayer.Conv2D(512, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = klayer.Conv2D(512, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = klayer.Dropout(0.5)(conv4)
    pool4 = klayer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = klayer.Conv2D(1024, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = klayer.Conv2D(1024, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = klayer.Dropout(0.5)(conv5)

    up6 = klayer.Conv2D(512, 2, activation="tanh", padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(drop5))
    merge6 = klayer.concatenate([drop4, up6], axis=3)
    conv6 = klayer.Conv2D(512, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = klayer.Conv2D(512, 3, activation="tanh", padding='same', kernel_initializer='he_normal')(conv6)

    up7 = klayer.Conv2D(256, 2, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv6))
    merge7 = klayer.concatenate([conv3, up7], axis=3)
    conv7 = klayer.Conv2D(256, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = klayer.Conv2D(256, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv7)

    up8 = klayer.Conv2D(128, 2, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv7))
    merge8 = klayer.concatenate([conv2, up8], axis=3)
    conv8 = klayer.Conv2D(128, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = klayer.Conv2D(128, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv8)

    up9 = klayer.Conv2D(64, 2, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv8))
    merge9 = klayer.concatenate([conv1, up9], axis=3)
    conv9 = klayer.Conv2D(64, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = klayer.Conv2D(64, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = klayer.Conv2D(2, 3, activation=keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = klayer.Conv2D(1, 1, activation="sigmoid")(conv9)

    out_layer = klayer.UpSampling2D(size=actual_pooling)(conv10)

    layers = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, out_layer]

    model = tf.keras.Model(inputs=inputs, outputs=layers if give_intermediate else out_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=bce_logdice_loss,
                  metrics=[metrics.binary_accuracy, mod_jaccard, dice_coef])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_orig(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4,
              give_intermediate=False, main_activation="relu", k_initializer="he_normal", zero_weight=0.5):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv1 = klayer.Conv2D(64, 3, activation=main_activation, padding='same', kernel_initializer=k_initializer)(
        scaled_inputs)
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
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=create_weighted_binary_crossentropy(zero_weight=zero_weight, one_weight=1 - zero_weight),
                  metrics=["accuracy"])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def smet(learning_rate, pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4, give_intermediate=False, main_activation=None, k_initializer="he_normal", zero_weight = 0.5):
    """Our custom small-net"""
    inputs = klayer.Input(input_size)
    conv3 = klayer.Conv2D(1, 1, activation=main_activation, use_bias=True, padding='same', kernel_initializer=k_initializer)(inputs)
    conv3 = klayer.LeakyReLU()(conv3)
    layers = [conv3]

    model = tf.keras.Model(inputs=inputs, outputs=layers if give_intermediate else conv3)

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=keras.losses.MSE,
                  metrics=[metrics.binary_accuracy, mod_jaccard])

    # model.summary()
    # all_layers_output = keras.backend.function([model.layers[0].input],
    #           [l.output for l in model.layers[1:]])

    return model


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

    img_input = klayer.Input(shape=input_shape)
    nb_channels = growth_rate * 2

    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')

    # Initial convolution layer
    x = klayer.Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1),
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

    x = klayer.BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = klayer.Activation('relu')(x)
    x = klayer.GlobalAveragePooling2D()(x)

    x = klayer.Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
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
        x = klayer.Concatenate(axis=-1)(x_list)
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
        x = klayer.BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = klayer.Activation('relu')(x)
        x = klayer.Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = klayer.Dropout(dropout_rate)(x)

    # Standard (BN-ReLU-Conv)
    x = klayer.BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = klayer.Activation('relu')(x)
    x = klayer.Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Dropout
    if dropout_rate:
        x = klayer.Dropout(dropout_rate)(x)

    return x


def convolution_block_xc(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = klayer.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = klayer.BatchNormalization()(x)
    if activation == True:
        x = klayer.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = klayer.LeakyReLU(alpha=0.1)(blockInput)
    x = klayer.BatchNormalization()(x)
    blockInput = klayer.BatchNormalization()(blockInput)
    x = convolution_block_xc(x, num_filters, (3,3) )
    x = convolution_block_xc(x, num_filters, (3,3), activation=False)
    x = klayer.Add()([x, blockInput])
    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """

    x = klayer.BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = klayer.Activation('relu')(x)
    x = klayer.Conv2D(int(nb_channels * compression), (1, 1), padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = klayer.Dropout(dropout_rate)(x)

    x = klayer.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def UXception(down_sampling=1, learning_rate=1e-4):

    backbone = Xception(input_shape=(128, 128, 3),weights='imagenet',include_top=False)
    input_layer = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[121].output
    conv4 = klayer.LeakyReLU(alpha=0.1)(conv4)
    pool4 = klayer.MaxPooling2D((2, 2))(conv4)
    pool4 = klayer.Dropout(0.1)(pool4)
    
     # Middle
    convm = klayer.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = klayer.LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = klayer.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = klayer.concatenate([deconv4, conv4])
    uconv4 = klayer.Dropout(0.1)(uconv4)
    
    uconv4 = klayer.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = klayer.LeakyReLU(alpha=0.1)(uconv4)
    
    deconv3 = klayer.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = klayer.concatenate([deconv3, conv3])    
    uconv3 = klayer.Dropout(0.1)(uconv3)
    
    uconv3 = klayer.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = klayer.LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = klayer.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = klayer.ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = klayer.concatenate([deconv2, conv2])
        
    uconv2 = klayer.Dropout(0.1)(uconv2)
    uconv2 = klayer.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = klayer.LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = klayer.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = klayer.ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = klayer.concatenate([deconv1, conv1])
    
    uconv1 = klayer.Dropout(0.1)(uconv1)
    uconv1 = klayer.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = klayer.LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = klayer.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = klayer.Dropout(0.1)(uconv0)
    uconv0 = klayer.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = klayer.LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = klayer.Dropout(0.1/2)(uconv0)
    output_layer = klayer.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = Model(input_layer, output_layer)
    #model.name = 'u-xception'

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=bce_logdice_loss,
                  metrics=[metrics.binary_accuracy, dice_coef])

    return model
