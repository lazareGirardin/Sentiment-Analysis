"""
************ AML PROJECT: SENTIMENT ANALYSIS ON MOVIE REVIEWS ******************

	This function trains a convolutional neural network over movie reviews.
	The reviews are split in words and tokenized, and matched to a lexicon integer entry
	in order to feed integers values to the network.

	The CNN is trained using Keras, which is based on TensorFlow (or Theano if chosen).

	This code explores the use of pre-trained word vectorization (GloVe)from all words of wikipedia.
	- Vectors can be found: https://nlp.stanford.edu/projects/glove/

	The part matching words and the pre-trained vectors is inspirated from Keras blog
	https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

		Input: 
			- NbFilters: Number of filters of convolution layers (and hidden layer)
			- EMBEDDING_DIM: dimension of GloVe vectors
			- kernel_size: size of convolution filters
			- dropout_rate: portion of nodes to ignore between certain layers to avoid overfitting
			- vocab_size: size of the vocabulary to consider (lexicon of vocab_size most frequent only)
			- name: name of the output
			- saveData: save accuracies at each epochs in Data/history/name.npy
			- saveModel: save model after training at Data/models/name.h5

		Output:
			- validation accuracy
			- training accuracy
			- accuracies during training if saveData = TRUE
			- model weights if saveModel = TRUE

		Example of use:

			GloVe(  nbFilters=128, EMBEDDING_DIM=200, kernel_size=3, dropout_rate = 0.25, 
			   		vocab_size=15000, name='dummy', saveData=True, saveModel=False )
	
	Authors: Hector Parmantier and Lazare Girardin

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from keras.models import load_model
#from keras.optimizers import SGD
from keras import backend as K

import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight
from nltk.tokenize import RegexpTokenizer

def GloVe( nbFilters=128, EMBEDDING_DIM=200, kernel_size=3, dropout_rate = 0.25, 
		   vocab_size=15000, name='dummy', saveData=True, saveModel=False ):


	nbClass = 5

	# ********* DATA LOAD AND PRE-PROCESS ********************

	x_train, x_test, y_train, y_test, y_int, y_true, embedding_matrix, max_words, dict_length =  pre_process_data(vocab_size, EMBEDDING_DIM)

	# ****** NEURAL NET DEFINITION AND TRAINING *********************

	model = create_architecture( dict_length, EMBEDDING_DIM, embedding_matrix, 
								 max_words, dropout_rate, nbFilters, kernel_size, nbClass)

	# Compute class weight as dictionary to inform Keras of the unbalance classes
	class_w = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)
	# Train the model
	history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=8, batch_size=32, class_weight=class_w, verbose=1)

	# ******** SAVE DATA *******************
	val_acc = history.history['val_categorical_accuracy']
	train_acc = history.history['categorical_accuracy']
	
	if saveData:
		path = 'Data/history/'+name
		save_name = path+'_AccVal.npy'
		np.save(save_name, val_acc)
		save_name = path+'_AccTrain.npy'
		np.save(save_name, train_acc)

	if saveModel:
		model_path = 'Data/models/'+name+'.h5'
		model.save(model_path)

	y_pred = model.predict_classes(x_test, verbose=1)
	cmat = confusion_matrix(y_true, y_pred)
	# Clear buffer from Keras verbose mode
	print('\n')
	# Print the confusion matrix of the classifier
	print(cmat)
	# Print the weighted f1-measure of the predictions
	print(f1_score(y_true, y_pred, average='weighted'))

	return val_acc, train_acc

def create_architecture(dict_length, EMBEDDING_DIM, embedding_matrix, max_words, dropout_rate, nbFilters, kernel_size, nbClass):
	"""
		Create a convolutional neural network with parameters:
		Input:
			- dict_length:		length of dictionary (used in the input shape)
			- EMBEDDING_DIM:	Dimension of word GloVe vectors 
			- embedding matrix: matrix matching word's unique integer values to its GloVe representation
			- max_words:		maximum length of one sentence (all other sequences are 0-padded to this length)
			- dropout_rate: 	portion of nodes to ignore between certain layers to avoid overfitting
			- NbFilters:		Number of conv. filters
			- kernel_size: 		size of convolution filters
			- nbClass:			Number of classes for the task	
		Output:
			- model:			Keras sequential neural net architecture	

	"""
	
	# Create the model
	model = Sequential()
	model.add(Embedding(dict_length + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
						input_length=max_words, trainable=False))
	model.add(Dropout(dropout_rate))

	#model.add(Conv1D(128, 3, border_mode='valid', activation='relu'))
	model.add(Conv1D(nbFilters, kernel_size, border_mode='valid', activation='relu'))
	
	model.add(GlobalMaxPooling1D())

	model.add(Dropout(dropout_rate))

	model.add(Dense(nbFilters))
	model.add(Dropout(dropout_rate))
	model.add(Activation('relu'))

	# Eventually add a second hidden layer
	model.add(Dense(nbFilters, activation='relu'))
	model.add(Dropout(dropout_rate))

	model.add(Dense(nbClass, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])
	print(model.summary())

	return model


def pre_process_data(vocab_size, EMBEDDING_DIM):
	"""
		This function load and pre-process the datas
		Input:
			- vocab_size: 		size of the vocabulary to consider (lexicon of vocab_size most frequent only)
			- EMBEDDING_DIM:	Dimension of word GloVe vectors 

		Output:
			- x_train: 			input train sequence of unique integers mapping words to lexicon
			- x_test:			test sequence of unique integers mapping words to lexicon
			- y_train: 			categorical train labels of x_train
			- y_test:   		categorical test labels of x_test
			- y_int:			integer train labels of x_train
			- y_true:			integer test labels of x_test
			- embedding_matrix: matrix matching word's unique integer values to its GloVe representation
			- max_words:		maximum length of one sentence (all other sequences are 0-padded to this length)
			- dict_length: 		size of the lexicon	

	"""

	# Load the data from csv files
	tr_f = './Data/train.tsv'
	te_f = './Data/test.tsv'
	train = pd.DataFrame.from_csv(tr_f, sep='\t')
	test = pd.DataFrame.from_csv(te_f, sep='\t')

	full = pd.concat([train, test])

	# Tokenize the sentences
	tokenizer = Tokenizer(nb_words=vocab_size)
	tokenizer.fit_on_texts(full["Phrase"])
	# Lexicon indices
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	dict_length = len(word_index)

	# Load the GloVe pre-trained vectors
	embeddings_index = {}
	GLOVE_DIR = 'Data/glove.6B/'
	glove_name = 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'
	f = open(os.path.join(GLOVE_DIR, glove_name))
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	#print('Found %s word vectors.' % len(embeddings_index))

	# Create a matrix matching word indices and the corresponding GloVe vectors
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		# None existing words stay 0
		if embedding_vector is not None:
			# Store the vectors if they exist
			embedding_matrix[i] = embedding_vector

	# ***** SET CREATION AND RANDOMIZATION ************************
	tr_ratio = 0.7

	# Transform sentences in sequence of integer corresponding to hash value of dict
	train_sequences = tokenizer.texts_to_sequences(train["Phrase"])
	x = sequence.pad_sequences(train_sequences)
	# Get labels and switch to categorical representation
	y = np.asarray(train["Sentiment"])
	labels = np_utils.to_categorical(y)
	# max sentence length
	max_words = x.shape[1]

	# Train/test split
	idx = np.random.permutation(np.arange(x.shape[0]))
	ll = int(tr_ratio*x.shape[0])

	x_train = x[idx[:ll]]
	y_train = labels[idx[:ll]]
	y_int = y[idx[:ll]]

	x_test = x[idx[ll:]]
	y_test = labels[idx[ll:]]
	y_true = y[idx[ll:]]

	return x_train, x_test, y_train, y_test, y_int, y_true, embedding_matrix, max_words, dict_length

	

	