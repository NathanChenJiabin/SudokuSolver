#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 16:29:34 2018

@author: jiabin CHEN

Ce programme s'occupe d'entrainer un modele de reseau neurone simple.
Vu que notre mission est de connaitre les chiffres de sodoku qui est en general claire sans bruit,
donc ce n'est pas necessaire d'utliser CNN(VGG16 ou ResNet etc) ou RNN(LSTM etc).

  
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Generate batches of tensor image data with real-time data augmentation. 
#The data will be looped over (in batches) indefinitely.
# we multiply the data by 1.0/255
train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(#Takes the path to a directory, and generates batches of augmented/normalized data
    'train',  # this is the target directory.  It should contain one subdirectory per class
    classes=list(map(str, range(10))), #optional list of class subdirectories
    color_mode='grayscale', # one color channel
    target_size=(28,28), #The dimensions to which all images found will be resized.
    batch_size=128, # size of the batches of data
    class_mode='categorical') #"categorical" will be 2D one-hot encoded labels

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    'valid',  # this is the target directory
    classes=list(map(str, range(10))),
    color_mode='grayscale',
    target_size=(28,28),
    batch_size=128,
    class_mode='categorical')
#Found 9140 images belonging to 10 classes of train.
#Found 1020 images belonging to 10 classes of valid.

# Define our model
# The Sequential model is a linear stack of layers
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# Before training a model, need to configure the learning process, 
# which is done via the compile method. It receives three arguments: an instance of optimizer, a loss function and a list of metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Fits the model on data generated batch-by-batch by a Python generator.
model.fit_generator(
        train_generator,
        steps_per_epoch=5120,#9140/128
                             #Total number of steps (batches of samples) to yield from generator before declaring
                             #one epoch finished and starting the next epoch. 
                             #It should typically be equal to the number of samples of your dataset divided by the batch size. 
        epochs=20, #Integer, total number of iterations on the data
        validation_data=validation_generator,
        validation_steps=512) # 1020/128

# Save model and weights
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('model.h5', overwrite=True)

