# This is a neural network built in Keras which learns to predict wether someone would survive on the Titanic or not

# Importing the neccessary libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
# load dataset
dataframe = pd.read_csv("ANN/Classification/TitanicSurvival/titanic.csv",  header = 0, )

# Get dummy values for the sex collumn
dfDummies = pd.get_dummies(dataframe['Sex'], prefix = 'category')
dataframe = pd.concat([dataframe, dfDummies], axis=1)

# Remove the name and non dummy values of sex from the data
del dataframe['Name']
del dataframe['Sex']

# Convert the dataframe into a numpy array
dataset = dataframe.values

# Split the data into inputs (X) and targets (Y)
X = dataset[:,1:8].astype(float)
Y = dataset[:,0].astype(float)

#Standardize the input data
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# define model
def model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=7, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model through 5 epochs but not more in order to prevent overfitting
model = model()
history = model.fit(x_train, y_train, epochs = 5, batch_size = 16, validation_data = (x_test, y_test)) 

# Plot the increase in accuracy on train and test data
plt.plot(history.history['acc'], label = "training accuracy")
plt.plot(history.history['val_acc'], label = "test accuracy")
plt.legend()
plt.show()

# Evaluate the model on unseen data; already done when training so not neccessary ( accuracy = 81.5% )
model.evaluate(x_test, y_test)









