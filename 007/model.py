import os
import sqlite3
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping

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

train_size = int(len(x) * 0.8)

train_x = x[:train_size]
test_x = x[train_size:]

train_y = y[:train_size]
test_y = y[train_size:]

num_labels = 357
vocab_size = 15000
batch_size = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_x)

x_train = tokenizer.texts_to_matrix(train_x, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_x, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_test = encoder.transform(test_y)

model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=30,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[early_stop])

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])

text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print('Actual label:' + test_y[i])
    print("Predicted label: " + predicted_label)

# creates a HDF5 file 'my_model.h5'
model.save(current_file + '/my_model.h5')

# Save Tokenizer i.e. Vocabulary
with open(current_file + '/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
