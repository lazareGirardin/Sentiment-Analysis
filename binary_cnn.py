from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import os
import pickle



def load_bin_data(naive=False):
    #load binarized data
    if naive:
        train = pd.DataFrame.from_csv("./Data/binary_naive.csv", sep='\t', encoding='utf-8')
    else:
        train = pd.DataFrame.from_csv("./Data/binary_2removed.csv", sep='\t', encoding='utf-8')
    return train["Phrase"], train["Sentiment"]


def glove_embedding(phrase, sentiment):
    tr_ratio = 0.8

    #Embedding construction
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(phrase)

    #Integer encoding of sentences
    train_sequences = tokenizer.texts_to_sequences(phrase)

    #Lexicon indices
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    #Load the GloVe pre-trained vectors
    embeddings_index = {}
    GLOVE_DIR = './Data/glove.6B/'
    EMBEDDING_DIM = 200
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    #Create matrix matching word indices and corresponding GloVe vectors
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	# None existing words stay 0
    	if embedding_vector is not None:
    		# Store the vectors if they exist
    		embedding_matrix[i] = embedding_vector

    #Create train and test data
    X = sequence.pad_sequences(train_sequences)
    y = np.asarray(sentiment)
    max_words = X.shape[1]

    #random train/test split
    idx = np.random.permutation(np.arange(X.shape[0]))
    tr_size = int(tr_ratio*X.shape[0])

    X_train = X[idx[:tr_size]]
    y_train = y[idx[:tr_size]]

    X_test = X[idx[tr_size:]]
    y_test = y[idx[tr_size:]]

    return (X_train, y_train), (X_test, y_test), (len(word_index), embedding_matrix)


def run_model(naive=False):
    max_features = 5000
    batch_size = 32
    embedding_dims = 200
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 5
    #sentiment_column = "Naive"


    print("Loading Data...")
    phrase, sentiment = load_bin_data(naive)
    (X_train, y_train), (X_test, y_test), (word_index_len, embedding_matrix) = glove_embedding(phrase, sentiment)

    print("Build Model...")
    model = Sequential()
    #Emedding: TO REMOVE
    model.add(Embedding(word_index_len + 1, embedding_dims, weights=[embedding_matrix], input_length=X_train.shape[1], trainable=False))

    model.add(Dropout(0.2))
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    y_pred = model.predict_classes(X_test, verbose=1)
    cmat = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(cmat)
    print(f1)
    return {"f1-score": f1, "history": history.history}



# columns = ["Naive", "Vader", "VaderOnPhrase"]
# results = {}
# for c in columns:
#     print("=============================== "+c+" ====================================")
#     results[c] = run_model(c)
#
# with open('assignment_performances.pickle', 'wb') as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

run_model(naive=True)
#with open('no2_assignment_performances.pickle', 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
