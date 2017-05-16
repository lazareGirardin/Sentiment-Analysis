from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
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



def cnn():

	method = 'stem'

	top_words = 10000
	size_embedding = 50
	max_words = 30
	balance = True
	t_ratio = 0.8

	batch = 32
	epochs = 10
	filters = 250
	kernel_size = 3
	hidden_dims = 250

	nb_class = 5

	x, y = load_dict(method, top_words)

	x_train, y_train, x_test, y_test, y_train_int, y_test_int = create_sets(x, y, t_ratio, max_words, balance)

	class_w = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)

	model = Sequential()
	model.add(Embedding(top_words, size_embedding, input_length = max_words))
	model.add(Dropout(0.2))

	model.add(Conv1D(filters, kernel_size,  border_mode='valid',
											activation='relu'))
	model.add(GlobalAveragePooling1D())

	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	model.add(Dense(nb_class, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['categorical_accuracy'])
	
	print(model.summary())

	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch, class_weight=class_w, verbose=1)

	acc_train = model.evaluate(x_train, y_train, verbose=0)[1]
	acc_test = model.evaluate(x_test, y_test, verbose=0)[1]
	y_pred = model.predict_classes(x_test, verbose = 0)
	cmat = confusion_matrix(y_test_int, y_pred)
	f1 = f1_score(y_test_int, y_pred, average=None)

	print(acc_train)
	print(acc_test)
	print(f1)
	print(cmat)

	import pdb; pdb.set_trace()