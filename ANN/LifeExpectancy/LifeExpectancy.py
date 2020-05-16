#This is a neural network made in Keras for predicting Life expectancy in various countries

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

#Import the dataset (If oyu want to implement this yourself you have to change the import path)
dataset = pd.read_csv("D:\Coding\AI\Projects\DeepL\ANN\LinearRegression\LifeExpectancy\Life Expectancy Data.csv",)
#Get dummy values for the countries (neural network only takes in numbers not words)
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
#Drop NaN values
dataset = dataset.dropna()

#Split the dataset into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Describe the data
train_stats = train_dataset.describe()
train_stats.pop("Life expectancy2")
train_stats = train_stats.transpose()

#Seperate the labels (life expectancy) from the input variables
train_labels = train_dataset.pop('Life expectancy2')
test_labels = test_dataset.pop('Life expectancy2')

#Initialize the standard scaler and use in to standardize the inputs
sc = StandardScaler()
normed_train_data = sc.fit_transform(train_dataset)
normed_test_data = sc.transform(test_dataset)

#Building the model
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

#Setting the number of epochs and initializing training
EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

#Evaluate the trained model on unseen test data (mean absolute error = 1.53)
results = model.evaluate(normed_test_data, test_labels, batch_size = 32)
print(results)

#Plot the decrease in loss (error) over epochs (learns fast and performs well on test data)
plt.plot(history.history['loss'])
plt.show()
