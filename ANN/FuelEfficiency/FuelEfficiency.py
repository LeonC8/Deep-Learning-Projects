#Predicting fuel efficiency of cars base on their specifications with a neural network in Keras
#The network predicts fuel efficiency based on the number of cylinders, the displacement, horsepower, weight of the car, the acceleration of the car, the model year and the origin country

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

#Get the dataset
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep = " ", skipinitialspace=True)
dataset = raw_dataset.copy()

#Drop NaN values in the dataset
dataset = dataset.dropna()
#Map origin to numbers based on country (neural network can only take in numbers)
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

#Split the dataset into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Get statistics in order to normalize the dataset
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

#Seperate the labels (fuel efficiency) from the inputs (specifications)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


#Normalize the data in order for the network to train faster
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0,2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0,2),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

  return model

model = build_model()

#Set number of epochs and train the keras model, store the history of the training
EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

#Evaluate the model on the test set (val_mean_absolute_error = 2.079)
results = model.evaluate(normed_test_data, test_labels, batch_size = 16)
print(results)

#Plot the decrease in loss over epochs (learns fast and performs well on test data)
plt.plot(history.history['val_loss'])
plt.show()