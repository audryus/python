import collections
import os
import sqlite3
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import RepeatVector, TimeDistributed, ActivityRegularization
import nltk

current_file = os.path.abspath(os.path.dirname(__file__))

connection = sqlite3.connect('conversation.db')
c = connection.cursor()
query = "SELECT * FROM conversation"

result = c.execute(query).fetchall()

Questions = []
Answers = []
Idx = []
i = 0
for row in result:
    Questions.append(row[0].lower())
    Answers.append(row[1].lower())
    Idx.append(i)
    i += 1

# Creating Vocabulary
counter = collections.Counter()

# QnAdata = pd.DataFrame(list(zip(Idx, Questions, Answers)))
QnAdata = Questions + Answers
for i in range(len(QnAdata)):
    # for word in nltk.word_tokenize(QnAdata.iloc[i][2]):
    for word in nltk.word_tokenize(QnAdata[i]):
        counter[word] += 1

word2idx = {w: (i+1) for i, (w, _) in enumerate(counter.most_common())}

idx2word = {v: k for k, v in word2idx.items()}

idx2word[0] = "PAD"

vocab_size = len(word2idx)+1

print(vocab_size)


def encode(sentence, maxlen, vocab_size):
    indices = np.zeros((maxlen, vocab_size))
    for i, w in enumerate(nltk.word_tokenize(sentence)):
        if i == maxlen:
            break
        indices[i, word2idx[w]] = 1
    return indices


def decode(indices, calc_argmax=True):
    if calc_argmax:
        indices = np.argmax(indices, axis=-1)
    return ' '.join(idx2word[x] for x in indices)


question_maxlen = 10

answer_maxlen = 20


def create_questions(question_maxlen, vocab_size):
    question_idx = np.zeros(
        shape=(len(Questions), question_maxlen, vocab_size))
    for q in range(len(Questions)):
        question = encode(Questions[q], question_maxlen, vocab_size)
        question_idx[q] = question
    return question_idx


quesns_train = create_questions(
    question_maxlen=question_maxlen, vocab_size=vocab_size)


def create_answers(answer_maxlen, vocab_size):
    answer_idx = np.zeros(shape=(len(Answers), answer_maxlen, vocab_size))
    for q in range(len(Answers)):
        answer = encode(Answers[q], answer_maxlen, vocab_size)
        answer_idx[q] = answer
    return answer_idx


answs_train = create_answers(
    answer_maxlen=answer_maxlen, vocab_size=vocab_size)

n_hidden = 128

question_layer = Input(shape=(question_maxlen, vocab_size))

encoder_rnn = LSTM(n_hidden, dropout=0.2,
                   recurrent_dropout=0.2)(question_layer)

repeat_encode = RepeatVector(answer_maxlen)(encoder_rnn)

dense_layer = TimeDistributed(Dense(vocab_size))(repeat_encode)

regularized_layer = ActivityRegularization(l2=1)(dense_layer)

softmax_layer = Activation('softmax')(regularized_layer)

model = Model([question_layer], [softmax_layer])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Model Training

quesns_train_2 = quesns_train.astype('float32')

answs_train_2 = answs_train.astype('float32')

model.fit(quesns_train_2, answs_train_2, batch_size=32,
          epochs=10, validation_split=0.05)

# Model prediction

ans_pred = model.predict(quesns_train_2[0:3])
print('Question: ', quesns_train_2[0:3])
print(decode(ans_pred[0]))
print(decode(ans_pred[1]))
