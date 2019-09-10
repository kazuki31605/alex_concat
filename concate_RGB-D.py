import keras
from keras import backend as K, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import save_model, load_model
from keras.layers import BatchNormalization, Embedding, Concatenate, Maximum, Add
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, merge
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Activation, Dropout, Conv1D, Conv2D, Reshape, Lambda
import sys

ROWS = 224
COLS = 224
CHANNELS = 3
label_list = []
image_list = []
# クラスの個数を指定
CLASSES = 9
batch = 32
epoch = 100

rgb_label_list = []
d_label_list = []

rgb_img_list = []
d_img_list = []

train_dir = 'C:/Users/hash/PycharmProjects/keras_deep/RGB-D/test_sukunai/'

# path以下の*ディレクトリ以下の画像を読み込む。
for train_path in os.listdir(train_dir):
    for dir in os.listdir('{}{}'.format(train_dir, train_path)):
        # print('{}{}/{}'.format(train_dir, train_path, dir))
        for file in os.listdir('{}{}/{}'.format(train_dir, train_path, dir)):
            if str(train_path) == "depth":
                d_label_list.append(int(dir) - 1)
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(train_dir, train_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                d_img_list.append(img)


            elif str(train_path) == "rgb":
                img = img_to_array(
                    load_img('{}{}/{}/{}'.format(train_dir, train_path, dir, file), target_size=(ROWS, COLS, CHANNELS)))
                rgb_img_list.append(img)

Y = to_categorical(d_label_list)


input_img_rgb = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_rgb")
input_img_d = Input(shape=(ROWS, COLS, CHANNELS), name="input_tensor_d")

#print(input_img_rgb)

cnn1 = Sequential()
cnn1 = Conv2D(96, 11, strides=(4, 4), input_shape=(ROWS, CLASSES, CHANNELS))(input_img_rgb)
cnn1 = Conv2D(256, 5)(cnn1)
cnn1 = Conv2D(384, 3)(cnn1)
cnn1 = Conv2D(384, 3)(cnn1)
cnn1 = Conv2D(256, 3)(cnn1)
cnn1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn1)
cnn1 = Dense(4096)(cnn1)
cnn1 = Dropout(0.5)(cnn1)
cnn1 = Dense(4096)(cnn1)


cnn2 = Sequential()
cnn2 = Conv2D(96, 11, strides=(4, 4), input_shape=(ROWS, COLS, CHANNELS))(input_img_d)
cnn2 = Conv2D(256, 5)(cnn2)
cnn2 = Conv2D(384, 3)(cnn2)
cnn2 = Conv2D(384, 3)(cnn2)
cnn2 = Conv2D(256, 3)(cnn2)
cnn2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn2)
cnn2 = Dense(4096)(cnn2)
cnn2 = Dropout(0.5)(cnn2)
cnn2 = Dense(4096, activation='relu', name='relu_g')(cnn2)

network = Concatenate()([cnn1, cnn2])
network = Dense(4096, activation='relu', name="last_dense")(network)
network = Flatten()(network)

# classification layer
network = Dense(CLASSES, activation='softmax', name='softmax')(network)

model = Sequential()
model = Model(inputs=[input_img_rgb, input_img_d], outputs=network)

##学習しない層の追加
print("学習しない層を表示")

for layer in model.layers[:9]:
    print(layer)
    layer.trainable = False

model.compile(optimizer=Adam(lr=1e-4, decay=1e-6),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

model.summary()  # モデルの表示

rgb_img_list = np.array(rgb_img_list)
d_img_list = np.array(d_img_list)

#print(len(rgb_img_list))

# 学習を実行。10%はテストに使用。
fit = model.fit(([rgb_img_list, d_img_list]), Y, batch_size=batch, epochs=epoch, validation_split=0.1)

model.save('./my_model_RGB_D_10.h5')
model.save_weights('./my_model_weights_RGB_D_10.h5')
