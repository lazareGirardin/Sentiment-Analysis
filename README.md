# Sentiment-Analysis
Sentiment Analysis on movie reviews. The aim is for a neural network to classify reviews left by users.

Workload (approximation):
  - 40% coding
  - 60% Analysis of methods and results

Dataset: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data (Reviews labeled in 5 classes, from Rottentomatoes)

(Eventually use IMDB dataset)

Pre-Processing in juypter-notebook Exploration.ipynb:
  - Keep only full sentences
  - Binarize Sentiment (1 or 0)
  - Tokenize sentences
  - Remove stopwords and similar (***** CHANGE THIS! KEEP IMPORTANT ONES *********)
  - ...

To Do:
  - Choose vocabulary size (tf-id? "by-hand" select few good features?...)
  - simple word2Vec, GloVe, ...  pre-processing
  - Compare performance with different pre-processing
  - Compare performance with different algorithms (simple perceptron, Neural Nets, basic ML algo?)
  - Best performance: upload on Kaggle to compare with existing solutions?

Preprocessing:
   - Tokenizer, without stopwords : 15273 unique words
   - Tokenizer and Stemmer: 10493 unique words!

### Binary Classification task
Sentiment Analysis on a binarized version of the RottenTomatoes dataset.

Two binarization strategies:
- Naive: Every reviews with y <= 2 mapped to 0 otherwise 1.
- Ambiguous reviews removed: Removal of every reviews where y = 2 (neutral).
Otherwise, f(y) = 0 if y < 2 and f(y) if y > 2.

Classification Model:
- pre-processing: Tokenization, GloVe embedding
- Convolutional Neural Network

Related files:
- binary_mapping_01.ipynb, binary_mapping_02.ipynb:
ipython notebooks used to visualize and map the reviews sentiments into binary values.
- Data/binary_naive.csv, Data/binary_2removed.csv:
csv files containing the two binarized datasets after mapping.
- binary_cnn.py:
python script training and running the CNN model on either one of the latter dataset.
See comment of scripts for execution.
