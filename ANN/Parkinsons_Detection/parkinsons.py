#This is a neural network built in Keras which learns to predict wether someone has Parkinsons disease based on many factors

#Importing the needed libraries
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
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("ANN\Classification\ParkinsonsDetection\parkinsons.csv",  header = 0, )

#Convert the pandas dataframe into a numpy array
dataset = dataframe.values
X = dataset[:,1:23].astype(float)
Y = dataset[:,23].astype(float)

#Standardize the input data
sc = StandardScaler()
X = sc.fit_transform(X)

#Split the training into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# define model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(2048, input_dim=22, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Initialize the model and train it
model = baseline_model()
history = model.fit(x_train, y_train, epochs = 20, batch_size = 16,) 

# Evaluate the model on unseen test data (accuracy = 92% )
results = model.evaluate(x_test, y_test)
print(results)

# Plot the increase in accuracy over time 
plt.plot(history.history['acc'])
plt.show()
