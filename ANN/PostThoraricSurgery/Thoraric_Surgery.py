#This is a neural network built in Keras which learns to predict the likelihood of a patient dying withing one year after a surgery

# Import the necessary libraries
import pandas
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
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("ANN\Classification\PostThoraricSurgery\ThoraricSurgery1.csv", header = 0, )
dataset = dataframe.values
X = dataset[:,0:-1]
Y = dataset[:,-1].astype(str)

#Standardize the inputs
x_sc = StandardScaler()
X = x_sc.fit_transform(X)

# Encode string input data to integers
ohe1 = OneHotEncoder()
ohe1.fit(X)
X = ohe1.transform(X)

#Encode the labels to integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, encoded_Y, test_size = 0.2)

# define model
def baseline_model():
    # create model
    model = Sequential()

    model.add(Dense(349, input_dim=349, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "sigmoid", kernel_initializer = "normal"))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model; in just one epoch it achieves a validation accuracy of 92%
model = baseline_model()
history = model.fit(x_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2)

#Evaluate the model on unseen test data (accuracy = 82% )
#The model does not generalize great to the test data; perhaps tweaking the architecture or getting more data could help
model.evaluate(x_test, y_test)
