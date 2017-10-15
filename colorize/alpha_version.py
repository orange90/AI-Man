# Copyright 2017 Jin Fagang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
"""
this is the first version of colorize
"""
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, gray2rgb
from skimage.io import imshow, imsave
import numpy as np
import os
import random
import tensorflow as tf


def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def train():
    image = img_to_array(load_img('assets/woman.jpg'))
    image = np.array(image, dtype=float)

    # get the Luminosity
    x = rgb2lab(image/255)[:, :, 0]
    # get the a, b as y
    y = rgb2lab(image/225)[:, :, 1:]

    x_batch = x.reshape(1, 400, 400, 1)
    y_batch = y.reshape(1, 400, 400, 2)

    model = build_model()
    model.fit(x_batch, y_batch, batch_size=1, epochs=100)
    model.save('alpha_model.h5')
    print("model train finished, let's predict an image.")


def colorize(image_path):
    # load model and predict a gray image
    if not os.path.exists(image_path):
        print('image {} not exist.'.format(image_path))
        exit(0)

    image = img_to_array(image_path)
    image = np.array(image, dtype=float)

    image = gray2rgb(image)
    image_lab = rgb2lab(image/225)
    image_l = image_lab[:, :, 0]

    model = build_model()
    model.load_weights('alpha_model.h5')
    y = model.predict(image_l)
    tmp = np.zeros((400, 400, 3))
    tmp[:, :, 0] =



