# This is a neural network built in Keras which learns to classify flowers based on the width and length of their sepals and petals.

# Importing the neccessary libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("ANN\Classification\Iris\iris.data", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, shuffle=True)

# define model
def model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
model = model()
history = model.fit(x_train, y_train, epochs= 500, validation_data=(x_test, y_test))

# Plot the increase in accuracy on train and test data
plt.plot(history.history['acc'], label = "training accuracy")
plt.plot(history.history['val_acc'], label = "test accuracy")
plt.legend()
plt.show()

# Evaluate the model on unseen data; already done when training so not neccessary ( accuracy = 96.67% )
model.evaluate(x_test, y_test)