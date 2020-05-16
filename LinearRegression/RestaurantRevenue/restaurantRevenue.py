import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
dataframe = pd.read_csv("D:/Coding/AI/Projects/DeepL/ANN/LinearRegression/RestaurantRevenue/Restaurant-Revenue-Prediction-master/train.csv",header=0)

dfDummiesCityGroup = pd.get_dummies(dataframe['CityGroup'], prefix = 'category')
dfDummiesCity = pd.get_dummies(dataframe['City'], prefix = 'category')
dfDummiesType = pd.get_dummies(dataframe['Type'], prefix = 'category')
dataframe = pd.concat([dataframe, dfDummiesCityGroup], axis=1)
dataframe = pd.concat([dataframe, dfDummiesCity], axis=1)
dataframe = pd.concat([dataframe, dfDummiesType], axis=1)
del dataframe['Id']
del dataframe['OpenDate']
del dataframe['City']
del dataframe['CityGroup']
del dataframe['Type']

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,2:78].astype(float)
Y = dataset[:,0].astype(float)

sc = StandardScaler()
X = sc.fit_transform(X)

sc2 = StandardScaler()
Y = sc2.fit_transform(Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

def baseline_model():
    model = Sequential()
    model.add(Dense(128, input_dim=75, activation='relu'))
    
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_squared_error", "mape"] )
    
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#model.fit
model = baseline_model()
model.fit(x_train, y_train, epochs = 50, shuffle = True, batch_size = 32)
score = model.predict(x_test, )
print(score)