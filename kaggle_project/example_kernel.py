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

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def plot_pixel_array(dataset, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


if __name__ == "__main__":
    data_pref = os.getcwd() + os.sep + "data" + os.sep + "dicom-images-train"

    im_height = 1024
    im_width = 1024
    num_imgs = 1377
    im_chan = 1

    print('reading input images and mask dataset.....')
    X_train = np.zeros((num_imgs, im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((num_imgs, im_height, im_width, 1), dtype=np.bool)

    for q, file_path in enumerate(glob(data_pref+"/*/*/*.dcm", recursive=True)):
        #print(file_path)
        df = pd.read_csv(os.getcwd() + "/data" + "/train-rle.csv", header=None, index_col=0)
        try:
            dataset = pydicom.dcmread(file_path)
        except:
            continue
        X_train[q] = np.expand_dims(dataset.pixel_array, axis=2)
        if df.loc[file_path.split('/')[-1][:-4],1] != -1:
            print(df)
            Y_train[q] = np.expand_dims(rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T, axis=2)
        else:
            Y_train[q] = np.zeros((1024, 1024, 1))

    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(X_train[0, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('mask')
    plt.imshow(Y_train[0, :, :, 0])
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('overlap')
    plt.imshow(label2rgb(Y_train[0, :, :, 0], image=X_train[0, :, :, 0], alpha=0.3))
    plt.axis('off')
    plt.show()
