# This is a neural network built in Keras which learns to recognize the 
# movement performed by human based on wathc accerelometer data.
# This model uses a simple neural network for multiclass classification and achieves a accuracy of 80%.
# The model is not very precise because it does not take into account time and only takes in current data into account.
# A better model for this would be a Recurrent Neural Network, specifically with LSTM cells which would take into account time.
# Nevertheless, a accuracy of 80% is suprising but would probably not work well in real life.
 
#  Import the necessary libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# load dataset (change the dataset path to replicate this on your computer)
dataframe = pandas.read_csv("D:/Coding/AI/Projects/DeepL/ANN/Classification/Activity recognition exp/Watch_accelerometer.csv", header=0)
#Convert the pandas dataframe into a numpy array
dataset = dataframe.values

# Split the dataset to input (X) and output (Y)
X = dataset[:,1:6].astype(float)
Y = dataset[:,8].astype(str)

# Convert action names to numbers (neural network can only take in numbers)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Standardize tha input data for easier and faster training
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=42)

# define model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=5, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Initialize the model and train it
model = baseline_model()
history = model.fit(X_train, y_train, epochs = 25, batch_size = 8, shuffle = True)

# Evaluate the model on the unseen test data
model.evaluate(X_test, y_test)

#Plot the increase in accuracy over epochs
plt.plot(history.history['acc'])
plt.show()