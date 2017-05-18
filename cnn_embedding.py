from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.models import load_model
#from keras.optimizers import SGD
from keras import backend as K

#from keras.datasets import imdb
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight

from data_utils import *

def cnn(folds):

	method = 'token'

	top_words = 15000
	size_embedding = 200
	max_words = 49
	balance = True
	t_ratio = 0.8

	batch = 32
	epochs = 8

	filters = 128
	#filters_list = [1, 10, 50, 100, 200, 250, 300, 350, 400, 500]
	
	kernel_size = 3
	#hidden_dims = 250
	#hidden_dims_list = [50, 100, 200, 300, 400, 500]
	hidden_dims_list = [128]

	nb_class = 5

	x, y = load_dict(method, top_words)

	acc = np.zeros((folds, 2, epochs))
	f1 = np.zeros((folds, nb_class))
	cmat = np.zeros((folds, nb_class, nb_class))

	test_length = len(hidden_dims_list)

	acc_mean = np.zeros((test_length, 2))
	acc_std = np.zeros((test_length, 2))
	f1_mean = np.zeros((test_length, nb_class))
	f1_std = np.zeros((test_length, nb_class))
	cmat_mean = np.zeros((test_length, nb_class, nb_class))
	cmat_std = np.zeros((test_length, nb_class, nb_class))

	for i, hidden_dims in enumerate(hidden_dims_list):
		print('testing convolution with {} hidden dims'.format(hidden_dims))
		for run in range(folds):
			print("# {} run".format(run))
			# Create a random split of the data
			x_train, y_train, x_test, y_test, y_train_int, y_test_int = create_sets(x, y, t_ratio, max_words, balance)
			class_w = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
			# Create the model with specified parameters
			model = create_struct(top_words, size_embedding, max_words, filters, kernel_size, hidden_dims, nb_class)
			history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch, class_weight=class_w, verbose=1)
			
			acc[run, 0] = history.history['val_categorical_accuracy']
			acc[run, 1] = history.history['categorical_accuracy']


			# Compute performance measure
			"""
			acc[fold, 0] = model.evaluate(x_train, y_train, verbose=0)[1]
			acc[fold, 1] = model.evaluate(x_test, y_test, verbose=0)[1]
			y_pred = model.predict_classes(x_test, verbose = 0)
			cmat[fold] = confusion_matrix(y_test_int, y_pred)
			f1[fold] = f1_score(y_test_int, y_pred, average=None)
			"""
		"""
		# Only keep mean and variance over folds
		acc_mean[i] = np.mean(acc, axis=0)
		acc_std[i] = np.std(acc, axis=0)
		f1_mean[i] = np.mean(f1, axis=0)
		f1_std[i] = np.std(f1, axis=0)
		cmat_mean[i] = np.mean(cmat, axis=0)
		cmat_std[i] = np.std(cmat, axis=0)
		"""
	np.save('Data/history/old_acc', acc)

	"""
	path = 'Data/structure/hidden_dims'

	np.save(path + '_acc_mean.npy', acc_mean)
	np.save(path + '_acc_std.npy', acc_std)
	np.save(path + '_f1_mean.npy', f1_mean)
	np.save(path + '_f1_std.npy', f1_std)
	np.save(path + '_cmat_mean.npy', cmat_mean)
	np.save(path + '_cmat_std.npy', cmat_std)

	print(acc_mean.shape)
	print(acc_std.shape)
	print(f1_mean.shape)
	print(cmat_shape.shape)
	"""

	#import pdb; pdb.set_trace()

def create_struct(top_words, size_embedding, max_words, filters, kernel_size, hidden_dims, nb_class):

	model = Sequential()
	model.add(Embedding(top_words, size_embedding, input_length = max_words))
	model.add(Dropout(0.2))

	model.add(Conv1D(filters, kernel_size,  border_mode='valid',
											activation='relu'))
	model.add(GlobalMaxPooling1D())

	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	model.add(Dense(nb_class, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])

	print(model.summary())
	return model