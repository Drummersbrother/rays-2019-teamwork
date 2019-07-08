"""Imports"""
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt

import os
import pickle
import traceback
from glob import glob

import tensorflow.keras.layers as klayer
import albumentations as alb

# from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization
from tensorflow.keras.models import Model
# from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.regularizers import l2

from keras_contrib.losses.jaccard import jaccard_distance as jaccard
from tensorflow.keras import metrics

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from tensorboard.plugins.beholder import Beholder

import json

from models import smet, unet, UXception, dice_coef

"""Load config"""
with open("C:\\rays-2019-teamwork\\kaggle_project\\config.json", mode="r") as f:
    config = json.load(f)

data_dir = config["data_dir"]
mask_csv_filename = config["mask_csv_filename"]

"""Create data loading utilities"""
def load_filep(f, down_sampling=8, mask_dir=""):
    # Load x-ray
    X = np.load(os.path.join(data_dir, "train_png", "preprocessed", os.path.split(f)[1]))

    # Load mask
    Y = np.load(os.path.join(mask_dir, "preprocessed", f.split(os.sep)[-1])[:-4]).astype(np.float32)  # .T

    # For UXception
    X = np.repeat(X, 3, axis=2)
    return X, Y

def augment_sample(X, Y):
    original_height, original_width = X.shape[:2]
    aug = alb.Compose([
        alb.OneOf([alb.RandomSizedCrop(min_max_height=(110, 127), height=original_height, width=original_width, p=0.3),
              alb.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
        alb.HorizontalFlip(p=0.5),
        #alb.OneOf([alb.GridDistortion(p=0.5)],p=0.2)
        ])

    augmented = aug(image=X, mask=Y)

    X = augmented['image']
    Y = augmented['mask']
    return X, Y

def siim_data_gen(filepaths, batch_size, dim=(1024, 1024), shuffle=True, mask_dir="", augment_data=True):
    """An infinite generator that gives X, Y of shape (batch_size, *dims, 1) (float32), from the given valid
    filepaths."""
    from random import shuffle as rshuffle
    
    down_sampling = 1024 // dim[0]

    # Actually generate an infinite number of epochs
    while True:
        # We generate batches for one epoch
        rshuffle(filepaths)
        for i in range(0, len(filepaths)+batch_size, batch_size):
            xs, ys = [], []
            for filepath in filepaths[i:i+batch_size]:
                X, Y = load_filep(filepath, down_sampling, mask_dir)

                if augment_data:
                    X, Y = augment_sample(X[:, :, 0], Y[:, :, 0])
                    X = np.expand_dims(X, axis=2)
                    X = np.repeat(X, 3, axis=2)
                    Y = np.expand_dims(Y, axis=2)

                xs.append(X)
                ys.append(Y)
            if len(xs) == 0:
                break
            yield np.array(xs), np.array(ys)


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
        a = np.asarray(Image.open(filepath[:-4] + ".png"))
        a = preprocess_image(a)
        X = (np.expand_dims(a, axis=2).astype(np.float32) + 1) / 2

        Y = np.load(os.path.join(self.mask_dir, filepath.split(os.sep)[-1])[:-4])  # .T
        # y_rle = self.rles[filepath.split(os.sep)[-1][:-4]]
        # Y = rle2mask(y_rle, *self.dim).T
        #  Y = np.reshape(Y, self.dim)
        Y = np.expand_dims(Y, axis=2).astype(np.float)
        #  Y = (X>0.5).astype(np.float)
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
        Image.open(filepath)
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

class BeholderCallback(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.beholder = Beholder(log_dir)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None, *args, **kwargs):
        pass

    def on_train_batch_end(self, batch, logs=None, *args, **kwargs):
        pass
        #K = keras.backend.backend
        #sess = keras.backend.get_session()
        #self.beholder.update(session=sess)
    
    def on_test_begin(*args, **kwargs):
        pass
    
    def on_test_end(*args, **kwargs):
        pass

    def on_test_batch_begin(*args, **kwargs):
        pass

    def on_test_batch_end(*args, **kwargs):
        pass

def preprocess_image(X: np.ndarray, down_sampling=8):
    down_widthed, down_heigted = X.shape[0] // down_sampling, X.shape[1] // down_sampling
    X = cv2.resize(X, (down_widthed, down_heigted), interpolation=cv2.INTER_CUBIC)
    X = np.expand_dims(X, axis=2).astype(np.float32)
    X = (X.astype(np.float32) - 128) / 128
    return X


def preprocess_mask(Y: np.ndarray, down_sampling=8):
    # mask = mask-128
    # mask = mask/128
    if len(Y.shape) == 1:
        Y = Y.reshape((1024, 1024))
    Y = Y.astype(np.float32)
    down_widthed, down_heigted = Y.shape[0] // down_sampling, Y.shape[1] // down_sampling
    Y = cv2.resize(Y, (down_widthed, down_heigted), interpolation=cv2.INTER_CUBIC)
    Y = np.expand_dims(Y, axis=2)
    return Y



"""Check that all data exists and is properly preprocessed"""
train_data_pref = data_dir + "train_png"
test_data_pref = data_dir + "test_png"

try:
    with open(data_dir + "valid_train_filepaths", mode="r") as f:
        valid_train_filepaths = f.read().split("\n")
    print("Loaded data from old list of valid data files")
except FileNotFoundError:
    print("Re-checking which data files are valid")
    rle_data = pd.read_csv(mask_csv_filename, header=None, index_col=0)
    valid_train_filepaths = [file_path[:-4] + ".npy" for file_path in
                             glob(os.path.join(train_data_pref, "*.png"), recursive=True)
                             if check_valid_datafile(file_path, rle_data)]
    valid_test_filepaths = [file_path[:-4] + ".npy" for file_path in
                            glob(os.path.join(test_data_pref, "*.png"), recursive=True)
                            if check_valid_datafile(file_path, rle_data, needs_label=False)]

    print(len(valid_test_filepaths), len(valid_train_filepaths))

    print("Converting all files into numpy-native format")

    def store_np_file(filepath, train=True):
        a = np.asarray(Image.open(filepath[:-4]+".png"))
        a = preprocess_image(a)
        np.save(os.path.join(data_dir, "train_png" if train else "test_png", "preprocessed", os.path.split(filepath)[1]), a)
    
    for inx, fp in enumerate(valid_train_filepaths):
        if inx % 100 == 0:
            print("Stored", inx + 1, "training images...")
        store_np_file(fp)
    print("Done storing training images")

    for inx, fp in enumerate(valid_test_filepaths):
        if inx % 100 == 0:
            print("Stored", inx + 1, "testing images...")
        store_np_file(fp)
    print("Done storing testing images")

    with open(data_dir + "valid_train_filepaths", mode="w") as f:
        f.write("\n".join(valid_train_filepaths))
    with open(data_dir + "valid_test_filepaths", mode="w") as f:
        f.write("\n".join(valid_test_filepaths))

    with open(mask_csv_filename, mode="r") as f:
        raw_rle = f.read()

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

"""Setup network training params"""

# Network and training params/config
dims = (128, 128)
n_epochs = 2000
batch_size = 16
img_downsampling = 8
learning_rate = 1e-4
num_train_examples = 2000
use_validation = True
validation_coeff = 0.1
retrain = True
cont_train = True
net_arch = "UXception"
monitor_weights = False

# The file in which trained weights are going to be stored
net_filename = f"data_aug_no_swa_{net_arch}-epochs_{n_epochs}-batchsz_{batch_size}-lr_{learning_rate}-downsampling_{img_downsampling}-numexamples_{num_train_examples}-valco_{validation_coeff}"

use_filepaths = pneumo_filepaths
num_validation_examples = int(num_train_examples * validation_coeff)
train_filepaths = use_filepaths[:-int(validation_coeff*len(use_filepaths)) if use_validation else len(use_filepaths)][:num_train_examples]
validation_filepaths = use_filepaths[-int(validation_coeff*len(use_filepaths)):][:num_validation_examples]

train_generator = siim_data_gen(train_filepaths, dim=dims, batch_size=batch_size,
                             mask_dir=os.path.join(data_dir, "train_png", "masks"))
if use_validation:
    validation_generator = siim_data_gen(validation_filepaths, batch_size=batch_size,
                                  mask_dir=os.path.join(data_dir, "train_png", "masks"))


def check_store_mask_file(raw_csv_data):
    key, rle_d = raw_csv_data.split(",")
    try:
        with open(os.path.join(data_dir, "train_png", "masks", key), mode="r") as f:
            pass
        with open(os.path.join(data_dir, "train_png", "masks", "preprocessed", key), mode="r") as f:
            pass
    except FileNotFoundError:
        rle_d = [x for x in rle_d.split() if x != ""]
        mask = rle2mask(rle_d, 1024, 1024).astype(np.bool).T
        with open(os.path.join(data_dir, "train_png", "masks", key), mode="wb") as f:
            np.save(f, mask)
        mask = preprocess_mask(mask)
        with open(os.path.join(data_dir, "train_png", "masks", "preprocessed", key), mode="wb") as f:
            np.save(f, mask)

    return os.path.join(data_dir, "train_png", "masks", key)


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
            print("Have processed", inx, "masks.")

    with open(os.path.join(data_dir, "valid_mask_paths"), mode="w") as f:
        f.write("\n".join(valid_mask_paths))

    print("Done computing valid mask paths")

print("Setup done!")

"""Train the network"""
keras.backend.clear_session()

# This is some weird debugging code, weird CUDA errors might appear if this code is deleted
#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Memory growth must be set before GPUs have been initialized
#    print(e)

print("Loading model", net_arch, "...")
model = locals()[net_arch](down_sampling=img_downsampling, learning_rate=learning_rate)
print("Done loading model.")

monitor_weights_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs:
    ([print(layer.get_weights()) for layer in model.layers]))

try:
    if retrain:
        raise Exception
    print("Loading pretrained network weights, from", os.path.join(data_dir + "models", net_filename))
    model.load_weights(os.path.join(data_dir + "models", net_filename))
    if cont_train:
        print("Continuing the training of the already saved model...")
        raise Exception
except Exception as e:
    print("Was not able to load model...")
    print("Training network!")

    from SWA import SWA
    callbacks = [
        #keras.callbacks.TensorBoard(log_dir=config["logdir"], histogram_freq=1, write_grads=True),
        #                            write_graph=True, write_images=True),
        #BeholderCallback(log_dir=config["logdir"]),
        keras.callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.2, patience=15, mode="min"),
        keras.callbacks.CSVLogger(os.path.join(data_dir, "histories", net_filename) + ".csv", append=True),
        #SWA(os.path.join(data_dir, "models", net_filename + "after_swa"), 80)
    ]
    if monitor_weights:
        callbacks.append(monitor_weights_callback)

    print("Fitting")
    
    data_gen = siim_data_gen(train_filepaths, batch_size, dim=(128, 128), mask_dir=os.path.join(data_dir, "train_png", "masks"), augment_data=True)
    if use_validation:
        validation_gen = siim_data_gen(validation_filepaths, batch_size, dim=(128, 128), mask_dir=os.path.join(data_dir, "train_png", "masks"), augment_data=False)

    try:
        please_stop = False
        """print(f"Loading {num_train_examples} samples for training")
        samples = []
        for i in range(num_train_examples):
            samples.append(next(data_gen))
            if i % 100 == 0:
                print("Loaded sample", i+1)
        print("Done loading samples, now processing them")
        x = np.asarray([e[0][0] for e in samples])
        y = np.asarray([e[1][0] for e in samples])"""
        
        print("Fitting network...")
        #model.fit(x, y, epochs=n_epochs)
        
        model.fit_generator(data_gen,
                  validation_data=validation_gen if use_validation else None,
                  validation_steps=num_validation_examples // batch_size,
                  steps_per_epoch=num_train_examples // batch_size,
                  epochs=n_epochs, use_multiprocessing=False,
                  callbacks=callbacks)
        
        print("Done fitting network.")

    except KeyboardInterrupt:
        please_stop = True
    except Exception as e:
        traceback.print_exc()
        please_stop = True
    finally:
        print("Saving model weights in", os.path.join(data_dir, "models", net_filename))
        model.save_weights(os.path.join(data_dir, "models", net_filename))
        print("Done saving model!")
        if please_stop:
            exit()
train_generator = siim_data_gen(train_filepaths, batch_size=batch_size,
                             mask_dir=os.path.join(data_dir, "train_png", "masks"))
prediction_model = locals()[net_arch] \
    (down_sampling=img_downsampling, learning_rate=learning_rate)#, give_intermediate=False)

prediction_model.load_weights(os.path.join(data_dir, "models", net_filename))

del model

"""Plot network predictions"""

print("Plotting predictions...")

import math

def plot_hidden_layer_activations(pred, layer_inx, imgs_per_row=None, filename=""):
    if not imgs_per_row:
        imgs_per_row = images_per_row
    # Displays the feature maps
    n_features = pred.shape[-1]  # Number of features in the feature map
    size = pred.shape[0]  # The feature map has shape (1, size, size, n_features).
    n_cols = math.ceil(n_features / imgs_per_row)  # Tiles the activation channels in this matrix
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
    plt.savefig(os.path.join(data_dir, "plots", "Layer_" + str(layer_inx + 1) + "_" + filename + figure_suffix),
                bbox_inches="tight")

def old_plot_code():
    print("also dummy")#for to_predict in []:#use_filepaths[num_train_examples:]:
    #    print("Plotting predictions/layers from", os.path.split(to_predict)[1], "...")
    #    x, y = load_filep(to_predict, down_sampling=8, mask_dir=os.path.join(data_dir, "train_png", "masks"))
    #
    #    pred_layers = prediction_model.predict(np.asarray([x]))
    #    x = x[:, :, 0]
    #
    #    sample_dice_coeff = K.get_session().run(dice_coef(y, pred_layers.squeeze(axis=0)))
    #
    #    plt.grid(False)
    #    plt.figure(figsize=(8, 8))
    #    plt.imshow(x.squeeze().astype(np.float32), aspect='auto', cmap='bone')
    #    plt.title("Input, dice_coef" + str(sample_dice_coeff))
    #    plt.savefig(os.path.join(data_dir, "plots", "Input_" + os.path.split(to_predict)[1] + figure_suffix),
    #                bbox_inches="tight")
    #    plt.close('all')
    #    #for layer_inx in layers_to_vis:
    #    #    print("\tOn layer nr.", layer_inx + 1)
    #    #    pred = pred_layers[layer_inx][0]
    #    #    plot_hidden_layer_activations(pred, layer_inx, filename=os.path.split(to_predict)[1])
    #    #    plt.close('all')
    #
    #    plt.figure(figsize=(8, 8))
    #    plt.title("Input, dice_coef" + str(sample_dice_coeff))
    #    x += 1
    #    x /= 2
    #    network_input = np.expand_dims(x, axis=2).repeat(3, axis=2) * 0.8
    #    ground_truth_mask = y.repeat(3, axis=2)
    #    ground_truth_mask[:, :, ::2] = 0
    #    pred_mask = pred_layers[0].repeat(3, axis=2)
    #    pred_mask[:, :, 1:] = 0
    #    
    #    superimposed = network_input + ground_truth_mask + pred_mask
    #    plt.imshow(superimposed)
    #
    #    plt.savefig(os.path.join(data_dir, "plots", "Summary_" + os.path.split(to_predict)[1] + figure_suffix),
    #                bbox_inches="tight")
    #    plt.show()
    print("dummy")

# Plot settings
plot_filepaths = train_filepaths
max_images = 32
grid_width = 4
threshold_best = 0.5

all_fps_to_plot = plot_filepaths[:max_images]
plt.close("all")
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
samples_to_plot = [load_filep(f, down_sampling=8, mask_dir=os.path.join(data_dir, "train_png", "masks")) for f in all_fps_to_plot]
xs, ys = [e[0] for e in samples_to_plot], [e[1] for e in samples_to_plot]
xs, ys = np.array(xs), np.array(ys)
preds_to_plot = prediction_model.predict(xs)

dice_intersections = []
dice_sums = []
dices = []

for idx, i in enumerate(all_fps_to_plot):
    print("Sub-plotting prediction from", os.path.split(i)[1], "...")
    x, y = load_filep(i, down_sampling=8, mask_dir=os.path.join(data_dir, "train_png", "masks"))
    x += 1
    x /= 2
    pred = preds_to_plot[idx]

    dice_intersection = y * pred.squeeze() * 2
    dice_sum = y.sum() + pred.squeeze().sum()
    dice_intersections.append(dice_intersection)
    dice_sums.append(dice_sum)
    pred_dice_coeff = K.get_session().run(dice_coef(y, pred.squeeze()))
    dices.append(pred_dice_coeff)

    ax = axs[int(idx / grid_width), idx % grid_width]
    ax.set_title(str(round(100*pred_dice_coeff, 2)) + r"% dice")
    network_input = x * 0.8
    ground_truth_mask = y.repeat(3, axis=2)
    ground_truth_mask[:, :, ::2] = 0
    pred_mask = pred.repeat(3, axis=2)
    pred_mask[:, :, 1:] = 0
    
    superimposed = network_input + ground_truth_mask + pred_mask
    ax.imshow(superimposed)
    ax.axis('off')
print("Total dice coeff:" + str(sum(dice_intersections)/(sum(dice_sums)+0.01)))
print("Mean dice coeff:" + str(round(100*(sum(dices)/len(dices)), 2)))
#plt.title("Mean dice coeff:" + str(round(100*(sum(dices)/len(dices)), 2)))
plt.show()