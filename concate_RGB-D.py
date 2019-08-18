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
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Activation, Dropout, Conv1D, Conv2D, Reshape, Lambda
import sys
from alex_rgb import alex_rgb
from alex_d import alex_d



ROWS = 224
COLS = 224
CHANNELS = 3
label_list = []
image_list = []
# クラスの個数を指定
CLASSES = 3
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



cnn1 = alex_rgb(input_img_rgb)
cnn2 = alex_d(input_img_d)

# leave only maximum features to eliminate null inputs
network = Concatenate()([cnn1, cnn2])
network = Dense(4096, activation='relu', name="last_dense")(network)

# classification layer
network = Dropout(0.5)(network)
network = Dense(CLASSES, activation='softmax', name='softmax')(network)

model = Sequential()
model = Model(inputs=[input_img_rgb, input_img_d], outputs=network)


model.compile(optimizer=Adam(lr=1e-4, decay=1e-6),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

model.summary()  # モデルの表示


# img_list = [bark_img_list, leaf_img_list, form_img_list]

# numpy配列に変更
rgb_img_list = np.array(rgb_img_list)
d_img_list = np.array(d_img_list)

#img_list = [bark_img_list, leaf_img_list, form_img_list]



# 学習を実行。10%はテストに使用。
fit = model.fit(([rgb_img_list, d_img_list]), Y,
                batch_size=batch, epochs=epoch, validation_split=0.1)

model.save('./my_model_RGB_D_10.h5')
model.save_weights('./my_model_weights_RGB_D_10.h5')

