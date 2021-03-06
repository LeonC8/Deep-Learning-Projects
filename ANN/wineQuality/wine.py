#Predict wine quality based on the presence of certain mollecules

#Import the necessary libraries
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

#Import the dataset into pandas (if you want to replicate this on your computer you have to change the file path)
dataset = pd.read_csv("D:\Coding\AI\Projects\DeepL\ANN\LinearRegression\wineQuality\winequality-red.csv",)
#Split the dataset into train and test sets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Describe the dataset
train_stats = train_dataset.describe()
train_stats.pop("quality")
train_stats = train_stats.transpose()

#Seperate the labels (Y) from the inputs(X)
train_labels = train_dataset.pop('quality')
test_labels = test_dataset.pop('quality')

#Standardize the inputs
sc = StandardScaler()
normed_train_data = sc.fit_transform(train_dataset)
normed_test_data = sc.transform(test_dataset)

#Build the model

#I used a pretty big model with 4 hidden layers, each containing 128 nodes which might not have been 
#necessary but since bigger models usually do not hurt performance I went with it.
#To reduce overfitting i used a dropout layer with 20% dropout after each layer

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

#Evaluate the model on unseen data (mean absolute percentage error = 7.69% )
#We have to take into account that the model had to learn from reviews 
#which were given by humans which are somewhat inconsistent.
#That is the reason why the model performs reasonably well but not exceptionally

results = model.evaluate(normed_test_data, test_labels, batch_size = 16)
print(results)

#Plot the decrease in loss (error) on validation data over epochs (learns fast and performs well on test data)
plt.plot(history.history['val_loss'])
plt.show()
