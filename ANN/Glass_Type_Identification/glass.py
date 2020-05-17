# This is a neural network built in Keras which learns to predict the type of glass based on various factors

# Import the neccessary libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# load dataset
dataframe = pandas.read_csv("ANN\Classification\GlassIdentification\glass.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:10].astype(float)
Y = dataset[:,10]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Standardize the input data
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data into train and test (small test split because we do not have a lot of data)
x_train, x_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.05, shuffle=True)

# define baseline model 
def model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=9, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
model = model()
history = model.fit(x_train, y_train, epochs= 1000, validation_data=(x_test, y_test))

# Plot the increase in accuracy on train and test data
plt.plot(history.history['acc'], label = "training accuracy")
plt.plot(history.history['val_acc'], label = "test accuracy")
plt.legend()
plt.show()

# Evaluate the model on unseen data; already done when training so not neccessary ( accuracy = 90.9% )
# The accuracy is not bad taking into account th fact that there is very little data and there are 6 classes
model.evaluate(x_test, y_test)