#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:33:42 2021

@author: Gabriel C.
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.0
x_test /= 255.0



classifier = Sequential()

def add(qtd_conv: int, qtd_dropout: float):
    global classifier
    for c in range(2):
        classifier.add(Conv2D(qtd_conv, (3, 3), activation='relu',
                          kernel_initializer='he_uniform', 
                          padding='same', input_shape=(32,32,3)))
    
        classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D((2,2)))
    classifier.add(Dropout(qtd_dropout))


add(32, 0.2)
add(64, 0.3)
add(128, 0.4)

classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu', 
                     kernel_initializer='he_uniform'))

classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation='softmax'))

opt = SGD(0.001, momentum=0.9)

classifier.compile(optimizer=opt, loss='categorical_crossentropy', 
              metrics=['accuracy'])

image_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                               horizontal_flip=True)

batch_size = 64

it_train = image_gen.flow(x_train, y_train, batch_size=batch_size)

steps = int(x_train.shape[0] / batch_size)

history = classifier.fit_generator(it_train, steps_per_epoch=steps,
                                   epochs=400,
                                   validation_data=(x_test, y_test),
                                   verbose=0)

_, acc = classifier.evaluate(x_test, y_test, verbose=0)

classifier.fit(x_train, y_train, epochs=100, batch_size=batch_size)

json_classifier = classifier.to_json()

with open('classifier_cifar10.json', 'w') as file:
    file.write(json_classifier)
    
classifier.save_weights('classifier_cifar10.h5')













