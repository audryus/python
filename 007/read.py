import os
import numpy as np
import pandas as pd
import sqlite3
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping

current_file = os.path.abspath(os.path.dirname(__file__))

# load our saved model
model = load_model(current_file + '/my_model.h5')

# load tokenizer
tokenizer = Tokenizer()
with open(current_file + '/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

connection = sqlite3.connect('conversation.db')
c = connection.cursor()
query = "SELECT * FROM conversation"

result = c.execute(query).fetchall()

x = []
y = []

for row in result:
    x.append(row[0])
    y.append(row[1])

x_data_series = pd.Series(
    ['She okay?', "you're gonna need to learn how to lie", "What good stuff?"])
x_tokenized = tokenizer.texts_to_matrix(x_data_series, mode='tfidf')

i = 0
for x_t in x_tokenized:
    prediction = model.predict(np.array([x_t]))
    predicted_label = y[np.argmax(prediction[0])]
    print("Predicted label: " + predicted_label)
    i += 1
