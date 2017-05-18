"""
************ AML PROJECT: SENTIMENT ANALYSIS ON MOVIE REVIEWS ******************

	This function trains a convolutional neural network over movie reviews.
	The reviews are split in words and tokenized, and matched to a lexicon integer entry
	in order to feed integers values to the network.

	The CNN is trained using Keras, which is based on TensorFlow (or Theano if chosen).

	This code explores the use of pre-trained word vectorization (GloVe)from all words of wikipedia.
	- Vectors can be found: https://nlp.stanford.edu/projects/glove/

	The part matching words and the pre-trained vectors is highly inspirated from Keras blog
	https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

	Example of use:

		GloVe(250, 300, 3, 0.25, 15000, 'struct1param1', True, True)
	
	authors: Hector Parmantier and Lazare Girardin

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

from data_utils import *

def GloVe( nbFilters=128, EMBEDDING_DIM=200, kernel_size=3, dropout_rate = 0.25, 
		   vocab_size=15000, name='dummy', saveData=True, saveModel=False ):

	tr_ratio = 0.8

	# ********* DATA PRE-PROCESSING ********************

	# Load the data from csv files
	tr_f = './Data/train.tsv'
	te_f = './Data/test.tsv'
	train = pd.DataFrame.from_csv(tr_f, sep='\t')
	test = pd.DataFrame.from_csv(te_f, sep='\t')

	full = pd.concat([train, test])

	# Tokenize the sentences
	tokenizer = Tokenizer(nb_words=vocab_size)
	tokenizer.fit_on_texts(full["Phrase"])
	# Transform sentences in sequence of integer corresponding to hash value of dict
	train_sequences = tokenizer.texts_to_sequences(train["Phrase"])
	# Lexicon indices
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

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
	print('Found %s word vectors.' % len(embeddings_index))

	# Create a matrix matching word indices and the corresponding GloVe vectors
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		# None existing words stay 0
		if embedding_vector is not None:
			# Store the vectors if they exist
			embedding_matrix[i] = embedding_vector


	# ***** SET RANDOMIZATION ************************

	# Create train and test data
	x = sequence.pad_sequences(train_sequences)
	y = np.asarray(train["Sentiment"])
	labels = np_utils.to_categorical(y)
	max_words = x.shape[1]

	idx = np.random.permutation(np.arange(x.shape[0]))
	
	ll = int(tr_ratio*x.shape[0])

	x_train = x[idx[:ll]]
	y_train = labels[idx[:ll]]
	y_int = y[idx[:ll]]

	x_test = x[idx[ll:]]
	y_test = labels[idx[ll:]]
	y_true = y[idx[ll:]]

	#print(x_train.shape)
	#print(y_train.shape)

	# ****** NEURAL NET DEFINITION AND TRAINING *********************

	nbClass = 5

	# Compute class weight as dictionary to inform Keras of the unbalance classes
	class_w = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)

	# Create the model
	model = Sequential()
	model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
						input_length=max_words, trainable=False))
	model.add(Dropout(dropout_rate))

	#model.add(Conv1D(128, 3, border_mode='valid', activation='relu'))
	model.add(Conv1D(nbFilters, kernel_size, border_mode='valid', activation='relu'))
	
	model.add(GlobalMaxPooling1D())

	model.add(Dropout(dropout_rate))

	#model.add(Flatten())
	#model.add(GlobalMaxPooling1D())
	model.add(Dense(nbFilters))
	model.add(Dropout(dropout_rate))
	model.add(Activation('relu'))

	model.add(Dense(nbClass, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])
	print(model.summary())

	# ******** SAVE DATA *******************
	
	history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=8, batch_size=32, class_weight=class_w, verbose=1)

	#print(history.history.keys())

	#import pdb;pdb.set_trace()
	
	if saveData:
		val_acc = history.history['val_categorical_accuracy']
		train_acc = history.history['categorical_accuracy']

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
	print(cmat)
	print(f1_score(y_true, y_pred))
	

	