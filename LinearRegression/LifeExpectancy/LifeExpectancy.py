from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical



dataset = pd.read_csv("C:/Users/Privat/Desktop/Coding/AI/Projects/TensorFlowTutorial/LinearRegression/Life Expectancy Data.csv",)
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Life expectancy")
train_stats = train_stats.transpose()


train_labels = train_dataset.pop('Life expectancy')
test_labels = test_dataset.pop('Life expectancy')

sc = StandardScaler()
sc2 = StandardScaler()

normed_train_data = sc.fit_transform(train_dataset)
normed_test_data = sc.transform(test_dataset)

def norm(x):
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0) +0.0001
    return (x - mean) / std 


def build_model():
  model = keras.Sequential([
    layers.Dense(32, activation='relu',  input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', ),
    layers.Dropout(0.2),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001, clipnorm=1)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse','mape'], )
  return model

model = build_model()

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

results = model.evaluate(normed_test_data, test_labels, batch_size = 32)
print(results)
