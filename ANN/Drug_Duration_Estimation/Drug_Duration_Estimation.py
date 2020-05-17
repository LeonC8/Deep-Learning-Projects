#This neural network built with Keras learns to estimate the amount of time a patient will have to be taking a certain medication

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
dataframe = pandas.read_csv("D:/Coding/AI/Projects/DeepL/ANN/Classification/datas/Datas.csv", header = 0, )
# Convert pandas dataframe to numpy array
dataset = dataframe.values

# Seperate the inputs (X) from the outputs (Y)
X = dataset[:,0:-1].astype(int)
Y = dataset[:,-1].astype(int)

#Standardize the inputs
x_sc = StandardScaler()
X = x_sc.fit_transform(X)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# define the model
def baseline_model():
    # create model
    model = Sequential()

    model.add(Dense(512, input_dim=112, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(1, kernel_initializer = "normal"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model

# Train the model
model = baseline_model()
history = model.fit(x_train, y_train, epochs = 50, batch_size = 32)

# Plot the decrease in mean absolute error on training data over epochs
plt.plot(history.history['mean_absolute_error'])
plt.show()

#Compute duration range ( max - min )
min_duration = y_test.min()
max_duration = y_test.max()
duration_range = max_duration - min_duration
# duration_range = 1918

# Evaluate the model on unseen test data 
result = model.evaluate(x_test, y_test)

mean_percentage_error = result[2] / duration_range
# Mean absolute percetange of 8.8% is achieved on unseen test data