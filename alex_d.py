from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from PIL import Image
import os
from keras.models import Sequential, Model
import glob
import numpy as np
from keras.optimizers import Adam
name_list = []
model = Sequential()

def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def alex_d(input_img):

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0), input_tensor=input_img, name='d_conv1')
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='d_pool1'))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1), name='d_conv2')
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='d_pool2'))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0), name='d_conv3')
    model.add(conv2d(384, 3, bias_init=1), name='d_conv4')
    model.add(conv2d(256, 3, bias_init=1), name='d_conv5')
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='d_pool3')
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層


    return model


