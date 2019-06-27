import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib as mpl
from skimage.color import label2rgb

import os
from glob import glob
import pydicom
from matplotlib import cm

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import sys
# sys.path.insert(0, '../input')
# This is one of the scripts included in the kaggle competition data
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
        X = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        Y = np.empty((self.batch_size, *self.dim), dtype=np.bool)

        # Generate data
        for i, filepath in enumerate(filepaths):
            # Store sample
            X[i,], Y[i] = self.load_filepath(filepath)

        return X, Y

    def load_filepath(self, filepath):
        train_rle_data = pd.read_csv(os.getcwd() + "/data" + "/train-rle.csv", header=None, index_col=0)
        dataset = pydicom.dcmread(file_path)
        X = np.expand_dims(dataset.pixel_array, axis=2)
        y_raw = str(train_rle_data.loc[file_path.split('/')[-1][:-4], 1])

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
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_shuffled_files = [self.filepaths[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(batch_shuffled_files)

        return X, Y


def check_valid_datafile(filepath, rle_df, needs_label=True):
    # We try to load each file in order to find which ones are invalid
    try:
        pydicom.dcmread(filepath, force=True)
    except Exception as e:
        print(e)
        print(f"Skipping loading of {filepath}, file didn't load as a DICOM correctly")
        return False

    if needs_label:
        try:
            str(rle_df.loc[filepath.split('/')[-1][:-4], 1])
        except:
            print(f"Skipping loading of {filepath}, file doesn't seem to exist")
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


if __name__ == "__main__":
    train_data_pref = data_dir + "dicom-images-train"
    test_data_pref = data_dir + "dicom-images-test"

    rle_data = pd.read_csv(os.getcwd() + "/data" + "/train-rle.csv", header=None, index_col=0)
    valid_train_filepaths = [file_path for file_path in
                             glob(train_data_pref + "/*/*/*.dcm", recursive=True)
                             if check_valid_datafile(file_path, rle_data)]
    valid_test_filepaths = [file_path for file_path in
                             glob(test_data_pref + "/*/*/*.dcm", recursive=True)
                             if check_valid_datafile(file_path, rle_data, needs_label=False)]

    print(len(valid_test_filepaths), len(valid_train_filepaths))
