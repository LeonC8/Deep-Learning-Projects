#This is a neural network built in Keras which learns to predict the house price in thousands of dollars

#Import the necessary libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

#Import the dataset into pandas
dataframe = pandas.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data", delim_whitespace=True, header=None)
#Convert the pandas dataframe into a numpy array
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
#Standardize the inputs
sc = StandardScaler()
X = sc.fit_transform(X)


# Define the model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse','mape'], )
    return model

#Initialize the model and train it
model = baseline_model()
results = model.fit(X,Y, epochs = 500)

#Show the decrease in mean absolute error over epochs (gets down to 1.45 mean absolute error)
plt.plot(results.history['mean_absolute_error'])
plt.show()