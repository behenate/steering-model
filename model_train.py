import os
import pandas as pd
import numpy as np
import copy
import random
from datetime import datetime
import matplotlib.pyplot as plt
import random
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout

from steering_utils import get_labels, Label
LABELS_FOLDER = 'labels/'
VAL_LABELS_FOLDER = 'val_labels/'

labels = get_labels(LABELS_FOLDER)
val_labels = get_labels(VAL_LABELS_FOLDER)
x_train, y_train = [],[]
x_valid, y_valid = [],[]
for label in labels:
    x_train.append(label.detection)
    y_train.append(label.angle)
for label in val_labels:
    x_valid.append(label.detection)
    y_valid.append(label.angle)

def create_model():
    model = Sequential()
    model.add(Dense(units=480, activation='relu',input_shape=(24,)))
    model.add(Dense(units=240, activation='relu'))
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=60, activation='relu'))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dense(units=7, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=Adam(lr=0.0001, decay=0.000001),loss='mse')
    return model
model = create_model()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
h = model.fit(x=x_train, y=y_train, batch_size=100, epochs=1000, validation_data=(x_valid, y_valid), shuffle=True, verbose=True)
model.save('steering_model.h5')