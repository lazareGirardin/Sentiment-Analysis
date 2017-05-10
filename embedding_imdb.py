from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, SpatialDropout2D
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from keras.datasets import imdb
import numpy as np
import os

from create_dict_full import *

# Import the data
def import_imdb():
	(x_train, y_train),(x_test, y_test) = imdb.load_data()

	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)

	return x_train, y_train, x_test, y_test

def try_binary():

	top_words = 15000
	x_filename = 'Data/x_%d' %top_words + '.npy'
	y_filename = 'Data/y_%d' %top_words + '.npy'

	if os.path.isfile(x_filename) and os.path.isfile(y_filename):
		x = np.load(x_filename)
		y = np.load(y_filename)
	else:
		x, y = create_data(top_words, False)
	
	x = sequence.pad_sequences(x)
	max_words = x.shape[1]

	x= x.reshape((x.shape[0], x.shape[1]))
	y = y >= 3

	t_ratio = 0.8
	tr_length = int(t_ratio*x.shape[0])

	# Add randomization here
	x_train = x[:tr_length]
	x_test = x[tr_length:]
	y_train = y[:tr_length]
	y_test = y[tr_length:]

	#import pdb; pdb.set_trace() 
	nb_class = 1

	model = Sequential()
	model.add(Embedding(top_words, 32, input_length=max_words))
	model.add(Conv1D(32, 12, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(2))
	model.add(Flatten())
	model.add(Dense(250, activation='relu'))
	model.add(Dense(250, activation='relu'))
	model.add(Dense(250, activation='relu'))
	model.add(Dense(nb_class, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy])
	print(model.summary())

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=2, batch_size=128, verbose=2)
	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print(scores)

def try_categ():

	top_words = 10000
	x_filename = 'Data/x_%d' %top_words + '.npy'
	y_filename = 'Data/y_%d' %top_words + '.npy'

	if os.path.isfile(x_filename) and os.path.isfile(y_filename):
		x = np.load(x_filename)
		y = np.load(y_filename)
	else:
		x, y = create_data(top_words, False)
	
	x = sequence.pad_sequences(x)
	max_words = x.shape[1]

	x= x.reshape((x.shape[0], x.shape[1]))

	t_ratio = 0.9
	tr_length = int(t_ratio*x.shape[0])

	# Add randomization here
	x_train = x[:tr_length]
	x_test = x[tr_length:]
	y_train = y[:tr_length]
	y_test = y[tr_length:]


	print(x_train.shape)

	#import pdb; pdb.set_trace() 
	nb_class = 5

	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	model = Sequential()
	model.add(Embedding(top_words, 64, input_length=max_words))
	model.add(Conv1D(64, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(2))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	#Output
	model.add(Dense(nb_class, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=3, batch_size=128, verbose=2)
	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))