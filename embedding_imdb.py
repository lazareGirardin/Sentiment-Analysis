from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

from keras.datasets import imdb
import numpy as np

# Import the data
def import_imdb():
	(x_train, y_train),(x_test, y_test) = imdb.load_data()

	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)

	return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = import_imdb()
print(len(x_train[0]))
print(len(x_train[1]))
dict_imdb = imdb.get_word_index()