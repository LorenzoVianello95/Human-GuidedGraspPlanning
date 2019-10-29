# CNN based on mnist model to evaluate if a grasp is good or bad at the exit of DexNet

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from util_functions import load_data

import numpy as np

batch_size = 64#128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols, channels = 100, 100, 4

train_data_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures"

# the data, split between train and test sets
x_train, y_train, x_eval, y_eval = load_data(train_data_path, 0.8)
#(x_train, y_train), (x_eval, y_eval) = mnist.load_data()

test_data_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures/test_set"
x_test, y_test, _, _ = load_data(test_data_path, 1)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_eval = x_eval.reshape(x_eval.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_eval = x_eval.reshape(x_eval.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols ,channels)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_eval = x_eval.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_eval.shape[0], 'eval samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_eval = keras.utils.to_categorical(y_eval, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_eval, y_eval))
score = model.evaluate(x_eval, y_eval, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

score_test= model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])
