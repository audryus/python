import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors.classification import KNeighborsClassifier


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):

        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    return set(filter_list)


def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i]
                               for i in vector[vector_index].indices})

    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


def filter_text(text):
    vec = TfidfVectorizer()
    desc_tfidf = vec.fit_transform(text)

    vocab = {v: k for k, v in vec.vocabulary_.items()}
    # Let's also filter some words out of the text vector we created
    filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

    # Use the list of filtered words we created to filter the text vector
    filtered_text = desc_tfidf[:, list(filtered_words)]
    return filtered_text.toarray()


current_file = os.path.abspath(os.path.dirname(__file__))

connection = sqlite3.connect('conversation.db')
c = connection.cursor()
query = "SELECT * FROM conversation"

result = c.execute(query).fetchall()

x = []
y = []

for row in result:
    x.append(row[0])
    y.append(row[1])

# Use vec's fit_transform method on the desc field

nb = MultinomialNB()

train_X, test_X, train_y, test_y = train_test_split(
    x, y, test_size=0.33, random_state=53)

vec = TfidfVectorizer()

train_tfidf = vec.fit_transform(train_X)
test_tfidf = vec.transform(test_X)
# Fit nb to the training sets
nb.fit(train_tfidf, train_y)

pred = nb.predict(test_tfidf)

score = metrics.accuracy_score(test_y, pred)

print(score)
