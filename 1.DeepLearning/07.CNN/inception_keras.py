# Source: https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import cv2 #python -m pip install opencv-python
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

num_classes = 10

label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def load_cifar10_data(img_rows, img_cols):
    # Load cifar10 training and test sets
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]])
    X_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_test[:, :, :, :]])

    X_train = X_train.astype('float16') / 255.0
    X_test = X_test.astype('float16') / 255.0

    # Transform targets to keras compatible format
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    print("X_train: {0}".format(X_train.shape))
    print("Y_train: {0}".format(Y_train.shape))
    print("X_test: {0}".format(X_test.shape))
    print("Y_test: {0}".format(Y_test.shape))

    return X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = load_cifar10_data(224, 224)

kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)

def inception_module(x,
                     filters_1x1,
                     filters_1x1_to_3x3,
                     filters_3x3,
                     filters_1x1_to_5x5,
                     filters_5x5,
                     filters_pool_1x1,
                     name=None):

    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters=filters_1x1_to_3x3, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters=filters_1x1_to_5x5, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = Conv2D(filters=filters_pool_1x1, kernel_size=(1, 1), padding='same', activation='relu',
                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool], axis=3, name=name)

    return output


input_layer = Input(shape=(224, 224, 3))

x = Conv2D(filters=64, kernel_size=(7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2',
           kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_1x1_to_3x3=96,
                     filters_3x3=128,
                     filters_1x1_to_5x5=16,
                     filters_5x5=32,
                     filters_pool_1x1=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_1x1_to_3x3=128,
                     filters_3x3=192,
                     filters_1x1_to_5x5=32,
                     filters_5x5=96,
                     filters_pool_1x1=64,
                     name='inception_3b')

x = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_1x1_to_3x3=96,
                     filters_3x3=208,
                     filters_1x1_to_5x5=16,
                     filters_5x5=48,
                     filters_pool_1x1=64,
                     name='inception_4a')


x1 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
x1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(units=1024, activation='relu')(x1)
x1 = Dropout(rate=0.7)(x1)
x1 = Dense(units=10, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_1x1_to_3x3=112,
                     filters_3x3=224,
                     filters_1x1_to_5x5=24,
                     filters_5x5=64,
                     filters_pool_1x1=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_1x1_to_3x3=128,
                     filters_3x3=256,
                     filters_1x1_to_5x5=24,
                     filters_5x5=64,
                     filters_pool_1x1=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_1x1_to_3x3=144,
                     filters_3x3=288,
                     filters_1x1_to_5x5=32,
                     filters_5x5=64,
                     filters_pool_1x1=64,
                     name='inception_4d')


x2 = AveragePooling2D(pool_size=(5, 5), strides=3)(x)
x2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(units=1024, activation='relu')(x2)
x2 = Dropout(rate=0.7)(x2)
x2 = Dense(units=10, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_1x1_to_3x3=160,
                     filters_3x3=320,
                     filters_1x1_to_5x5=32,
                     filters_5x5=128,
                     filters_pool_1x1=128,
                     name='inception_4e')

x = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_1x1_to_3x3=160,
                     filters_3x3=320,
                     filters_1x1_to_5x5=32,
                     filters_5x5=128,
                     filters_pool_1x1=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_1x1_to_3x3=192,
                     filters_3x3=384,
                     filters_1x1_to_5x5=48,
                     filters_5x5=128,
                     filters_pool_1x1=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(rate=0.4)(x)

x = Dense(units=10, activation='softmax', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')

model.summary()

initial_lrate = 0.01

def decay(epoch, steps=100):
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

lr_sc = LearningRateScheduler(decay, verbose=1)

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=True)

model.compile(
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    loss_weights=[1, 0.3, 0.3],
    optimizer=sgd,
    metrics=['accuracy']
)

epochs = 25

history = model.fit(
    x=X_train,
    y=[y_train, y_train, y_train],
    validation_data=(X_test, [y_test, y_test, y_test]),
    epochs=epochs, batch_size=256, callbacks=[lr_sc]
)