

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = "relu" ))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(64, 3, 3, activation = "relu" ))

classifier.add(Convolution2D(64, 3, 3, activation = "relu" ))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(128, 3, 3, activation = "relu" ))

classifier.add(Convolution2D(128, 3, 3, activation = "relu" ))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(512, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(43, activation='softmax'))

sgd = SGD(lr = 0.01, decay= 1e-6, momentum=0.9, nesterov = True)
classifier.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = [ "accuracy" ])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



training_set = train_datagen.flow_from_directory( 'C:/Users/Privat/Desktop/Coding/AI/Projects/DeepL/CNN/TrafficSigns/Images' ,
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')



classifier.fit_generator( training_set,
                     steps_per_epoch=39252,
                     epochs=30,
                     )

