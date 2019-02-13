import sys
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
import keras.layers 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras import models
from keras import optimizers
from PIL import Image

import matplotlib.pyplot as plt

num_classes = 10
epochs = 40
batch_size = 64
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') # for division
x_test = x_test.astype('float32')
x_train /= 255 # normalise
x_test /= 255
y_train = to_categorical(y_train, num_classes) #binary transfer
y_test = to_categorical(y_test, num_classes)

if sys.argv[1] == 'train':
	model = Sequential()
	model.add(keras.layers.convolutional.Conv2D(32,5,strides=(1, 1),padding='same',data_format=None,
		input_shape=(32,32,3))) # CNN layer 1
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #max pool layer
	model.add(Dropout(0.25))

	model.add(keras.layers.convolutional.Conv2D(10,5,strides=(1, 1),padding='same')) # CNN layer 2
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #max pool layer
	model.add(Dropout(0.25))

	model.add(keras.layers.core.Flatten()) #from multi to one
	model.add(keras.layers.core.Dense(120,activation=None, use_bias=True))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(keras.layers.core.Dense(84,activation=None, use_bias=True))
	model.add(Activation('tanh'))

	model.add(keras.layers.core.Dense(10,activation='softmax', use_bias=True))
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	path = './model.h5'
	model.save(path)
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

	result = model.evaluate(x_test, y_test, verbose=1)
	print('Test accuracy:', result[1])

elif sys.argv[1] == 'test':
	x = Image.open(sys.argv[2])
	x = np.array(x)
	x = np.resize(x, (1, 32, 32, 3))
	from keras.models import load_model
	model = load_model('model.h5')
	labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	y = model.predict(x)
	print(labels[np.argmax(y)])


