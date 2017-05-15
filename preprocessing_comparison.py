from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.models import load_model
#from keras.optimizers import SGD
from keras import backend as K

#from keras.datasets import imdb
import numpy as np
import os

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight

from data_utils import *

def perceptron_comparison(folds):

	TRAIN = 0
	TEST = 1
	epochs = 5
	batch = 32
	max_words = 30
	top_words = 10000
	size_embedding = 60
	nb_class = 5

	t_ratio = 0.7

	balance = True

	#methods_list = ['token', 'stopwords', 'selected_stopwords', 'stem', 'lemm', 'word2vec', 'glove', 'n_gram']
	methods_list = ['token']
	acc = np.zeros((folds, 2))
	f1 = np.zeros((folds, nb_class))
	cmat = np.zeros((folds, nb_class, nb_class))

	path = 'Data/result_matrices/'

	# For all methods that we want to compare
	for method in methods_list:
		print('Evaluation of the {} method'.format(method))
		# Load the corresponding dictionary
		x, y = load_dict(method, top_words)
		for fold in range(folds):
			print('\t fold # {}'.format(fold))
			# Create random datasets
			x_train, y_train, x_test, y_test, y_train_int, y_test_int = create_sets(x, y, t_ratio, max_words, balance)
			model = create_model(top_words, size_embedding, max_words, nb_class)
			class_w = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
			model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch, class_weight=class_w, verbose=1)
			acc[fold, TRAIN] = model.evaluate(x_train, y_train, verbose=0)[1]
			acc[fold, TEST] = model.evaluate(x_test, y_test, verbose=0)[1]
			y_pred = model.predict_classes(x_test, verbose = 0)
			cmat[fold, :, :] = confusion_matrix(y_test_int, y_pred)
			f1[fold, :] = f1_score(y_test_int, y_pred, average=None)
			print(acc[fold])
			print(f1[fold])
		name = path + method + '_acc.npy'
		np.save(name, acc)
		name = path + method + '_f1.npy'
		np.save(name, f1)
		name = path + method + '_cmat.npy'
		np.save(name, cmat)


	"""# Full Stopwords
				x, y = load_dict('stopwords', max_words)
			
				# Selected Stopwords
				x, y = load_dict('selected_stopwords', max_words)
			
				# Stemmer
				x, y = load_dict('stem', max_words)
			
				# Lemmatizer
				x, y = load_dict('lemm', max_words)
			
				# Word2Vec
				x, y = load_dict('word2vec', max_words)
			
				# GloVe
				x, y = load_dict('glove', max_words)
			
				# Bi-gram (on stemmer)
				x, y = load_dict('n_gram', max_words)"""


	return

def create_model(top_words, size_embedding, max_words, nb_class):

	model = Sequential()
	model.add(Embedding(top_words, size_embedding, input_length = max_words))
	model.add(GlobalAveragePooling1D())
	model.add(Dense(nb_class, activation='softmax'))	
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])
	return model