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

#from keras.datasets import imdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight
from nltk.tokenize import RegexpTokenizer

from data_utils import *

def GloVe():

	tr_ratio = 0.8

	tr_f = './Data/train.tsv'
	train = pd.DataFrame.from_csv(tr_f, sep='\t')
	te_f = './Data/test.tsv'
	test = pd.DataFrame.from_csv(te_f, sep='\t')

	full = pd.concat([train, test])

	tokenizer = Tokenizer(nb_words=15000)
	tokenizer.fit_on_texts(full["Phrase"])

	sequences = tokenizer.texts_to_sequences(train["Phrase"])
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	embeddings_index = {}
	GLOVE_DIR = 'Data/glove.6B/'
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	print('Found %s word vectors.' % len(embeddings_index))

	EMBEDDING_DIM = 100
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector


	x = sequence.pad_sequences(sequences)
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

	print(x_train.shape)
	print(y_train.shape)

	class_w = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)

	model = Sequential()

	model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_words, trainable=False))
	model.add(Dropout(0.2))

	model.add(Conv1D(128, 3, border_mode='valid', activation='relu'))

	model.add(GlobalMaxPooling1D())

	model.add(Dense(250))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])
	print(model.summary())


	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=32, class_weight=class_w, verbose=1)

	y_pred = model.predict_classes(x_test, verbose=1)
	cmat = confusion_matrix(y_true, y_pred)
	print(cmat)
	print(f1_score(y_true, y_pred, average=None))