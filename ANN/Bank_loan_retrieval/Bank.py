#This is a nural network which learns to predict wether someone will return their loan or not

#Import the necessary libraries
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
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("ANN/Classification/Bank/bank.csv", sep = ";", header = 0, )
#Convert the pandas dataframe into a numpy array
dataset = dataframe.values
# Split the data into input (X) and output (Y) variables
X = dataset[:,1:16].astype(str)
Y = dataset[:,16].astype(str)

# encode input values and output labels as numbers
ohe1 = OneHotEncoder()
ohe1.fit(X)
X = ohe1.transform(X)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, encoded_Y, test_size = 0.2)

# define model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=3651, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Initialize the model and train it
model = baseline_model()
history = model.fit(x_train, y_train, epochs = 10, batch_size = 32)

#Evaluate the trained model on unseen test data (accuracy on unseen data = 87.2% )
results = model.evaluate(x_test, y_test, batch_size = 12)
print(results)

#Plot the increase in accuracy on training data over epochs ( not fully representative of the actual real life accuracy )
plt.plot(history.history['acc'])
plt.show()