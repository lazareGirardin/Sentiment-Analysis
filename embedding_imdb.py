from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, SpatialDropout2D, LSTM, BatchNormalization
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

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight

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
	model.add(Embedding(top_words, 64, input_length=max_words))
	model.add(Conv1D(64, 12, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(2))
	model.add(Flatten())
	model.add(Dense(250, activation='relu'))

	model.add(Dense(nb_class, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=2, class_weight='auto', batch_size=128, verbose=1)
	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	y_pred = model.predict_classes(x_test, verbose=1)

	print('\n')
	print(f1_score(y_test, y_pred))

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

	t_ratio = 0.6
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
	y_true = y_test.copy() # Keep real values for test results
	y_test = np_utils.to_categorical(y_test)

	model = Sequential()
	model.add(Embedding(top_words, 128, input_length=max_words))
	model.add(Conv1D(128, 16, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(2))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(512, activation='relu'))

	#Output
	model.add(Dense(nb_class, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=3, batch_size=128, verbose=1)
	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	y_pred = model.predict(x_test, verbose=1)

	y_pred_class = np.argmax(y_pred, axis=1)


	print('\n')
	cmat = confusion_matrix(y_true, y_pred_class)
	print(cmat)
	print(f1_score(y_true, y_pred_class, average=None))

	import pdb; pdb.set_trace()


def try_LSTM():

	stemmer = True
	binary_class = False
	top_words = 8000

	x_filename = 'Data/x_%d' %top_words + '.npy'
	y_filename = 'Data/y_%d' %top_words + '.npy'

	if os.path.isfile(x_filename) and os.path.isfile(y_filename):
		x = np.load(x_filename)
		y = np.load(y_filename)
	else:
		x, y = create_data(top_words, binary_class, stemmer)
	
	x = sequence.pad_sequences(x)
	max_words = x.shape[1]

	x= x.reshape((x.shape[0], x.shape[1]))

	t_ratio = 0.7
	tr_length = int(t_ratio*x.shape[0])

	# Add randomization here
	x_train = x[:tr_length]
	x_test = x[tr_length:]
	y_train = y[:tr_length]
	y_test = y[tr_length:]

	class_w = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

	print(x_train.shape)

	#import pdb; pdb.set_trace() 
	nb_class = 5

	y_train = np_utils.to_categorical(y_train)
	y_true = y_test.copy() # Keep real values for test results
	y_test = np_utils.to_categorical(y_test)

	model = Sequential()
	model.add(Embedding(top_words, 64, input_length=max_words, trainable=True))
	model.add(Conv1D(64, 4, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(2))

	model.add(LSTM(64, return_sequences=False, dropout_W = 0.3, dropout_U = 0.3))
	model.add(BatchNormalization())
	#model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	#model.add(Dense(64, activation='relu'))

	# Add dropout
	model.add(Dropout(0.5))

	#Output
	model.add(Dense(nb_class, activation='softmax'))


	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	print(model.summary())

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=3, batch_size=128, class_weight=class_w,verbose=1)
	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	y_pred = model.predict(x_test, verbose=1)

	y_pred_class = np.argmax(y_pred, axis=1)


	print('\n')
	cmat = confusion_matrix(y_true, y_pred_class)
	print(cmat)
	print(f1_score(y_true, y_pred_class, average=None))
	print(f1_score(y_true, y_pred_class, average='weighted'))

	#import pdb; pdb.set_trace()