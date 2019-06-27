import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib as mpl
from skimage.color import label2rgb

import os
from glob import glob
#import pydicom
from matplotlib import cm

import tensorflow.keras.layers as klayer

import numpy as np

data_dir = os.getcwd() + os.sep + "data" + os.sep


class DataLoader(keras.utils.Sequence):
    def __init__(self, filepaths, batch_size=32, dim=(1024, 1024), shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.filepaths = filepaths
        self.indices = np.arange((len(self.filepaths)))
        self.shuffle = shuffle
        self.on_epoch_end()

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
        train_rle_data = pd.read_csv(os.path.join(os.getcwd(), "data", "train-rle.csv"), header=None, index_col=0)
        X = np.array(Image.open(filepath))
        X = np.expand_dims(X, axis=2)
        y_raw = str(train_rle_data.loc[filepath.split(os.sep)[-1][:-4], 1])
        if len(y_raw.split()) != 1:
            Y = np.expand_dims(rle2mask(y_raw, *self.dim).T, axis=2)
        else:
            Y = np.zeros((*self.dim, 1))
        return X, Y

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
    split_data = rle.split()
    array_data = []
    for x in split_data:
        try:
            array_data.append(int(x))
        except ValueError:
            pass
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


def plot_pixel_array(dataset, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


def unet(pretrained_weights=None, input_size=(1024, 1024, 1), down_sampling=4):
    """Almost directly taken from https://github.com/zhixuhao/unet. Modified to fit into memory"""
    inputs = klayer.Input(input_size)
    # Rescale to not take too much memory
    scaled_inputs = klayer.MaxPooling2D(pool_size=(down_sampling, down_sampling))(inputs)
    conv1 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(scaled_inputs)
    conv1 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = klayer.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = klayer.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = klayer.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = klayer.Dropout(0.5)(conv4)
    pool4 = klayer.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = klayer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = klayer.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = klayer.Dropout(0.5)(conv5)

    up6 = klayer.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(drop5))
    merge6 = klayer.concatenate([drop4, up6], axis=3)
    conv6 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = klayer.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = klayer.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv6))
    merge7 = klayer.concatenate([conv3, up7], axis=3)
    conv7 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = klayer.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = klayer.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv7))
    merge8 = klayer.concatenate([conv2, up8], axis=3)
    conv8 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = klayer.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = klayer.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        klayer.UpSampling2D(size=(2, 2))(conv8))
    merge9 = klayer.concatenate([conv1, up9], axis=3)
    conv9 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = klayer.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = klayer.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = klayer.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=klayer.UpSampling2D(size=(down_sampling, down_sampling))(conv10))

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    train_data_pref = data_dir + "train_png"
    test_data_pref = data_dir + "test_png"

    try:
        with open(data_dir + "valid_train_filepaths", mode="r") as f:
            valid_train_filepaths = f.read().split("\n")
        print("Loaded data from old list of valid data files")
    except FileNotFoundError:
        print("Re-checking which data files are valid")
        rle_data = pd.read_csv(os.path.join(os.getcwd(), "data", "train-rle.csv"), header=None, index_col=0)
        print(os.path.join(os.getcwd(), "data", "train-rle.csv"), rle_data)
        valid_train_filepaths = [file_path for file_path in
                                 glob(os.path.join(train_data_pref, "*.png"), recursive=True)
                                 if check_valid_datafile(file_path, rle_data)]
        valid_test_filepaths = [file_path for file_path in
                                glob(os.path.join(test_data_pref, "*.png"), recursive=True)
                                if check_valid_datafile(file_path, rle_data, needs_label=False)]

        print(len(valid_test_filepaths), len(valid_train_filepaths))

        with open(data_dir + "valid_train_filepaths", mode="w") as f:
            f.write("\n".join(valid_train_filepaths))
        with open(data_dir + "valid_test_filepaths", mode="w") as f:
            f.write("\n".join(valid_test_filepaths))

    train_generator = DataLoader(valid_train_filepaths, batch_size=1)

    unet_model = unet(down_sampling=4)

    unet_model.fit_generator(train_generator, epochs=1, use_multiprocessing=True)
