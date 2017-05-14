import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from keras.utils import np_utils
from keras.preprocessing import sequence

from collections import Counter
import operator

def load_dict(method, max_words, datas = None, to_load=True):
	
	if to_load:
		tr_f = './Data/train.tsv'
		datas = pd.DataFrame.from_csv(tr_f, sep='\t')

	# This variable indicates which column to take as data
	dataColumn = 'token'

	if method == 'token':
		datas['token'] = datas.apply(tokenize, axis=1)

	elif method == 'stopwords':
		datas['token'] = datas.apply(tokenize_stopwords, axis=1)

	elif method == 'selected_stopwords':
		datas['token'] = datas.apply(tokenize_selected, axis=1)

	elif method == 'stem':
		datas['token'] = datas.apply(tokenize, axis=1)
		datas['stem'] = datas.apply(stem_words, axis=1)
		dataColumn = 'stem'

	elif method == 'lemm':
		datas['token'] = datas.apply(tokenize, axis=1)
		datas['PoS'] = datas.apply(pos_tagging, axis=1)
		datas['lemm'] = datas.apply(lem_words, axis=1)
		dataColumn = 'lemm'

	#elif method == 'word2vec':

	#elif method == 'glove':

	#elif method == 'n_gram':

	else:
		print("Method name unknown!")

	# Create a dictionnary on the selected dataColumn
	words = []
	for i in range(datas.shape[0]):
		for word in datas[dataColumn].iloc[i]:
			words.append(word)

	counts = Counter(words)
	print('The dataset contains {} unique words'.format(len(counts)))

	# 0 is used for padding in keras, 1 is Out-of-Voc form
	OoV = 1
	# Total extra char used
	nb_extraChar = 2
	maxDictLength = max_words - nb_extraChar

	sorted_words = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
	word_dict = dict([ (sorted_words[i][0], i+nb_extraChar)for i in range(maxDictLength)])

	def words_to_dict(row):
		return [[word_dict[r] if (r in word_dict) else OoV] for r in row[dataColumn]]

	# X and Y corresponds to the dictionnary values and Sentiment values
	datas["Dict values"] = datas.apply(words_to_dict, axis=1)
	X = np.array(datas["Dict values"])
	Y = np.array(datas["Sentiment"])

	return X, Y


def create_sets(x, y, t_ratio, max_words):
	"""
		Create random partition of the data (ADD CLASS BALANCE ARGUMENT)
		Input:
			- x : datas
			- y : labels
			- t_ratio : train/test ratio

		Output:
			- x_train
			- y_train_c : categorical matrix of y_train
			- x_test
			- y_test_c :  categorical matrix of y_test
			- y_train :   integers values of y_train
			- y_test: 	  integers values of y_test
	"""


	# Pad all sequence with same length
	x = sequence.pad_sequences(x, max_words)
	x= x.reshape((x.shape[0], x.shape[1]))

	tr_length = int(t_ratio*x.shape[0])
	idx = np.random.permutation(np.arange(x.shape[0]))
	x_train = x[idx[:tr_length]]
	x_test = x[idx[tr_length:]]
	y_train = y[idx[:tr_length]]
	y_test = y[idx[tr_length:]]

	# Use caterogical matrix to train the network
	y_train_c = np_utils.to_categorical(y_train)
	y_test_c = np_utils.to_categorical(y_test)

	return x_train, y_train_c, x_test, y_test_c, y_train, y_test


def tokenize(df):
	# Tokenize and remove punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	english_sw = []
	tokens = tokenizer.tokenize(df['Phrase'])
	return [t.lower() for t in tokens if t.lower() not in (english_sw + ['rrb','lrb'])]

def tokenize_stopwords(df):
	# Tokenize and remove punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	english_sw = stopwords.words('english')
	tokens = tokenizer.tokenize(df['Phrase'])
	return [t.lower() for t in tokens if t.lower() not in (english_sw + ['rrb','lrb'])]

def tokenize_selected(df):
	# Tokenize and remove punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	# CHANGE HERE !
	english_sw = ['a', 'i', 'me', 'you', 'of']
	tokens = tokenizer.tokenize(df['Phrase'])
	return [t.lower() for t in tokens if t.lower() not in (english_sw + ['rrb','lrb'])]

def stem_words(row):
	eng_stemmer = SnowballStemmer('english')
	return [eng_stemmer.stem(word) for word in row['token']]

def pos_tagging(df):
	pos_tags = nltk.pos_tag(df['token'])
	return [(PoS[0], penn_to_wn(PoS[1])) for PoS in pos_tags]

def lem_words(row):
	w_lemmatizer = WordNetLemmatizer()
	return [[w_lemmatizer.lemmatize(word, tag) if tag else w_lemmatizer.lemmatize(word)] for (word, tag) in row['PoS']]

def is_noun(tag):
	return tag in ['NN', 'NNS', 'NNP', 'NNPS']
def is_verb(tag):
	return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def is_adverb(tag):
	return tag in ['RB', 'RBR', 'RBS']
def is_adjective(tag):
	return tag in ['JJ', 'JJR', 'JJS']
def penn_to_wn(tag):
	if is_adjective(tag):
		return wn.ADJ
	elif is_noun(tag):
		return wn.NOUN
	elif is_adverb(tag):
		return wn.ADV
	elif is_verb(tag):
		return wn.VERB
	return None
