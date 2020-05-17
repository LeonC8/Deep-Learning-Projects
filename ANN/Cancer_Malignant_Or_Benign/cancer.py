# This is a neural network built in Keras which learns to classify cancers as benign or malignant based on their properties

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
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("C:/Users/Privat/Desktop/Coding/AI/Projects/DeepL/ANN/Classification/WisconsinCancer/data.csv",  header = 0, )
dataframe.drop('id', axis = 1)
dataset = dataframe.values
X = dataset[:,2:32].astype(float)
Y = dataset[:,1].astype(str)

# encode label values as integers ( M and B to 0s and 1s)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, encoded_Y, test_size = 0.2)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16,  activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Train the model
model = baseline_model()
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2) 

# Plot the accuracy and validation accuracy change over epochs
plt.plot(history.history['acc'], label = "train_accuracy")
plt.plot(history.history['val_acc'], label = "validation_accuracy")
plt.legend()
plt.show()

# Evaluate the model on unseen test data; achieved 91 - 93% accuracy, 
# although given the nature of the algorithm we get different results almost every time
model.evaluate(x_test, y_test)

