import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import operator

oovf = 1

def load_train():
	tr_f = './Data/train.tsv'
	train = pd.DataFrame.from_csv(tr_f, sep='\t')

	#Keep only first  full sentence
	print("Keeping full sentences...")
	full = train.copy()
	full = full.groupby("SentenceId").apply(keep_first)

	print("Tokenizing...")
	full['Phrase tokenized'] = full.apply(tokenize_stopwords, axis=1)

	return full

def tokenize_stopwords(df):
	# Tokenize and remove punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	english_sw = stopwords.words('english')
	tokens = tokenizer.tokenize(df['Phrase'])
	#tokens = nltk.word_tokenize(df['Phrase'])
	return [t.lower() for t in tokens if t.lower() not in (english_sw + ['rrb','lrb'])] 

def keep_first(group):
	return pd.Series({"Phrase": group["Phrase"].iloc[0], "Sentiment": group["Sentiment"].iloc[0]})

def create_data(maxDictLength=0, binary_class=True):
	"""
		Creating a dictionary of the unique words in the train set ordered
		by their frequencies in the reviews.

	"""

	print("loading data...")
	datas = load_train()

	words = []
	for i in range(datas.shape[0]):
		for word in datas['Phrase tokenized'].iloc[i]:
			words.append(word)

	counts = Counter(words)
	print('The dataset contains {} unique words'.format(len(counts)))

	if maxDictLength == 0:
		maxDictLength = len(counts)

	sorted_words = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
	
	nb_extraChar = 2 # 0 is reserved for padding, 1 for Out Of Vocabulary forms
	word_dict = dict([ (sorted_words[i][0], i+nb_extraChar)for i in range(maxDictLength)])

	def words_to_dict(row):
		return [[word_dict[r] if (r in word_dict) else oovf] for r in row["Phrase tokenized"]]

	datas["Dict values"] = datas.apply(words_to_dict, axis=1)

	x = np.array(datas["Dict values"])
	y = np.array(datas["Sentiment"])

	if binary_class:
		y_bin = y.copy()
		y_bin[y>2]=1
		y_bin[y<=2]=0
		y = y_bin.copy()
		# With Booean:
		# y = np.array(datas.Sentiment >= 3)

	t_ratio = 0.9
	tr_length = int(t_ratio*x.shape[0])

	# Add randomization here
	x_train = x[:tr_length]
	x_test = x[tr_length:]
	y_train = y[:tr_length]
	y_test = y[tr_length:]

	return x_train, x_test, y_train, y_test




