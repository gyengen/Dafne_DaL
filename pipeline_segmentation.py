#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat Mar 21 16:22:26 2020

@author: abramo
"""

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
import tensorflow as tf
from keras.activations import softmax
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Concatenate, Lambda, ZeroPadding2D, Activation, Reshape
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from keras import optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
import json
import liver_seg as unet 
#import gamba as unet

IMAGE_DIR_PATH = '/Users/norbertgyenge/new_research/Liver/data/'
MASK_DIR_PATH = '/Users/norbertgyenge/new_research/Liver/label/'
BATCH_SIZE = 72

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

from datal import DaL

dataset = DaL(image_paths=image_paths,
              mask_paths=mask_paths,
              image_size=(432, 432),
              crop_percent=0.8,
              channels=(3, 1),
              augment=True,
              compose=False,
              seed=47)

ds = dataset.data_batch(batch_size=BATCH_SIZE, shuffle=True)

#train_ds, val_ds, test_ds = dataset.get_dataset_partitions_tf(72)

# Initialize the data queue

X_train = []

for images, masks in ds:

	for image in images:

		X_train.append(image / 255)

y_train = masks / 255


X_train = np.array(X_train)
y_train = np.array(y_train)


checkpoint_path="Weights_liver/weights_liver-{epoch:02d}-{loss:.2f}.hdf5" 

check=ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=10)

adamlr = optimizers.Adam(learning_rate=0.009765, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)

unetshallow=unet.unet()

#unetshallow.compile(loss=unet.weighted_loss, optimizer=adamlr, metrics=[unet.evaluation_metric])

unetshallow.compile(loss=unet.weighted_loss, optimizer=adamlr)

#unetshallow.compile(loss='sparse_categorical_crossentropy', optimizer=adamlr)

history = unetshallow.fit(X_train, y_train, epochs=5, validation_split = 0.1)



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['evaluation_metric'])
plt.plot(history.history['val_evaluation_metric'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
