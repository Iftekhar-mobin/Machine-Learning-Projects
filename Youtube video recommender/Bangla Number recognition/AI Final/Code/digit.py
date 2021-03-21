#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Convolutional Neural Networ
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import theano
import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
from keras.models import load_model
from matplotlib.pyplot import imshow
from keras.callbacks import History 
from keras.constraints import maxnorm
#classifier = load_model("my_model.h5")
# Initialising the CNN
classifier = Sequential()


################################################################
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
#classifier.add(Dropout(0.2))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
#classifier.add(Dropout(0.2))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
#classifier.add(Dropout(0.2))

classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(10, activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 10
                                                 )

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 10
                                            )
history = History()
classifier.fit_generator(training_set,
                         steps_per_epoch = 1110,
                         epochs =10,
                         callbacks=[history], 
                         validation_data = test_set,
                         validation_steps = 281)

#classifier.save("digit.h5")


# load model
classifier = load_model("digit.h5")

import matplotlib.pyplot as plt
get_2rd_layer_output = K.function([classifier.layers[0].input], [classifier.layers[2].output])
print(get_2rd_layer_output)
# list all data in history

print(history.history['acc'])


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summerize the history of loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import pydot
from keras.utils import plot_model
plot_model(classifier, to_file='model.png',show_shapes=True)
# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dl5.jpg', target_size = (64, 64))
#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

#model.save("my_model2.h5")
