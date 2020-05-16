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




dataset = pd.read_csv("C:/Users/Privat/Desktop/Coding/AI/Projects/TensorFlowTutorial/LinearRegression/winequality-red.csv",)




train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("quality")
train_stats = train_stats.transpose()


train_labels = train_dataset.pop('quality')
test_labels = test_dataset.pop('quality')

sc = StandardScaler()
normed_train_data = sc.fit_transform(train_dataset)
normed_test_data = sc.transform(test_dataset)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation='relu',  input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.24),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.24),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.24),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.24),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse','mape'], )
  return model

model = build_model()

EPOCHS = 5000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

results = model.evaluate(normed_test_data, test_labels, batch_size = 16)
print(results)
