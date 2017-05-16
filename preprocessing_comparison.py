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
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight

from data_utils import *

def perceptron_comparison(folds):

	TRAIN = 0
	TEST = 1
	epochs = 10
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
			# Create a new model
			model = create_model(top_words, size_embedding, max_words, nb_class)
			# Native Keras solution for unbalnaced classes
			class_w = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
			# Train the model
			model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch, class_weight=class_w, verbose=1)
			# Compute different score data: accuracy, f1, confusion matrix
			acc[fold, TRAIN] = model.evaluate(x_train, y_train, verbose=0)[1]
			acc[fold, TEST] = model.evaluate(x_test, y_test, verbose=0)[1]
			y_pred = model.predict_classes(x_test, verbose = 0)
			cmat[fold, :, :] = confusion_matrix(y_test_int, y_pred)
			f1[fold, :] = f1_score(y_test_int, y_pred, average=None)
			print(acc[fold])
			print(f1[fold])
		# Save the data to create a plot
		name = path + method + 'bal_acc.npy'
		np.save(name, acc)
		name = path + method + 'bal_f1.npy'
		np.save(name, f1)
		name = path + method +  'bal_cmat.npy'
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

def plot_results(folds):

	path = 'Data/result_matrices/'
	methods_list = ['token', 'stopwords', 'selected_stopwords', 'stem', 'lemm']
	colors = ['b', 'r', 'g', 'm', 'y']

	acc = np.zeros((len(methods_list), folds, 2))
	f1 = np.zeros((len(methods_list), folds, 5))
	cmat = np.zeros((len(methods_list), folds, 5, 5))

	for i, method in enumerate(methods_list):
		name = path + method + '_acc.npy'
		acc[i, :, :] = np.load(name)
		name = path + method + '_f1.npy'
		f1[i, :, :] = np.load(name)
		name = path + method + '_cmat.npy'
		cmat[i, :, :, :] = np.load(name)

	f1_mean_folds = np.mean(f1, axis = 1)
	f1_std_folds = np.std(f1, axis = 1)

	plt.figure(1)
	plt.title(" Pre-Processing Comparison", fontsize=30)
	for i in range(len(methods_list)):
		plt.errorbar(np.arange(5), f1_mean_folds[i], f1_std_folds[i], label=methods_list[i], color=colors[i])
	plt.legend(fontsize= 20)
	plt.xlabel("Classes", fontsize = 20)
	plt.ylabel("f1-measure", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()

	acc_mean = np.mean(acc, axis = 1)
	acc_std = np.std(acc, axis = 1)
	print(acc_std.shape)

	plt.figure(2)
	plt.errorbar(np.arange(len(methods_list)), acc_mean[:, 0], acc_std[:, 0], label='train')
	plt.errorbar(np.arange(len(methods_list)), acc_mean[:, 1], acc_std[:, 0], label='test')
	plt.xlabel("method", fontsize = 20)
	plt.ylabel("accuracy", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()

def plot_balnce(folds):
	path = 'Data/result_matrices/token'
	f1_balance = np.load(path+'bal_f1.npy')
	f1_unblance = np.load(path+'Unbalanced_f1.npy')

	plt.figure(2)
	plt.title(" Effect of skewed class distribution", fontsize=30)
	plt.errorbar(np.arange(5), np.mean(f1_balance, axis=0), np.std(f1_balance, axis=0), label='Undersampled class 2')
	plt.errorbar(np.arange(5), np.mean(f1_unblance, axis=0), np.std(f1_unblance, axis=0), label='Raw dataset')
	plt.legend(fontsize= 20)
	plt.xlabel("class", fontsize = 20)
	plt.xlim([-0.1,4.1])
	plt.ylabel("f1-measure", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()