import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.stem import SnowballStemmer
import operator

oovf = 1

def load_train(stemmer = False):
	tr_f = './Data/train.tsv'
	train = pd.DataFrame.from_csv(tr_f, sep='\t')

	#full = full.groupby("SentenceId").apply(keep_first)

	print("Tokenizing...")
	train['Phrase tokenized'] = train.apply(tokenize_stopwords, axis=1)
	
	if stemmer:
		train['stemmed'] = train.apply(stem_words, axis = 1)

	return train


def stem_words(row):
    # Add ~2% on class accuracy!
    eng_stemmer = SnowballStemmer('english')
    return [eng_stemmer.stem(word) for word in row["Phrase tokenized"]]


def tokenize_stopwords(df):
	# Tokenize and remove punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	#english_sw = stopwords.words('english')
	english_sw = []
	tokens = tokenizer.tokenize(df['Phrase'])
	#tokens = nltk.word_tokenize(df['Phrase'])
	return [t.lower() for t in tokens if t.lower() not in (english_sw + ['rrb','lrb'])] 

def keep_first(group):
	return pd.Series({"Phrase": group["Phrase"].iloc[0], "Sentiment": group["Sentiment"].iloc[0]})

def create_data(max_words=0, binary_class=False, stemmer = False):
	"""
		Creating a dictionary of the unique words in the train set ordered
		by their frequencies in the reviews.

	"""
	if stemmer:
		DataColumn = "stemmed"
	else:
		DataColumn = "Phrase tokenized"
	

	print("loading data...")
	datas = load_train(stemmer)

	words = []
	for i in range(datas.shape[0]):
		for word in datas[DataColumn].iloc[i]:
			words.append(word)


	counts = Counter(words)
	print('The dataset contains {} unique words'.format(len(counts)))

	nb_extraChar = 2 # 0 is reserved for padding, 1 for Out Of Vocabulary forms

	if max_words == 0:
		maxDictLength = len(counts)
	else:
		maxDictLength = max_words - nb_extraChar

	sorted_words = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
	
	word_dict = dict([ (sorted_words[i][0], i+nb_extraChar)for i in range(maxDictLength)])

	def words_to_dict(row):
		return [[word_dict[r] if (r in word_dict) else oovf] for r in row[DataColumn]]

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

	np.save('Data/x_%d' %max_words + '.npy',x)
	np.save('Data/y_%d' %max_words + '.npy', y)

	return x, y





