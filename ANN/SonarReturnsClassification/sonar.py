#This is a neural network built in Keras which learns to recognize wether the return of sonar signals were bounced of a metal or a rock

# Import the necessary libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load dataset
dataframe = pandas.read_csv("ANN\Classification\SonarReturns\sonar.csv", header=None)
# Convert dataframe to numpy array
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# Standardize the input data for easier training
sc = StandardScaler()
X = sc.fit_transform(X)

# Convert label strings to integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, encoded_Y, test_size = 0.2)

# define the model
def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# train the model
model = create_model()
history = model.fit(x_train, y_train, epochs = 65, validation_data = (x_test, y_test))

# Visualize the increase in accuracy over epochs
plt.plot(history.history['acc'], label = 'train_accuracy')
plt.plot(history.history['val_acc'], label = 'validation_accuracy')
plt.show()

# Evaluate the model on unseen test data (accuracy = 90% )
model.evaluate(x_test, y_test)
