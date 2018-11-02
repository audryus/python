import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

party = ['party']

current_file = os.path.abspath(os.path.dirname(__file__))

headers = ['infants',  'water',  'budget',  'physician',  'salvador',
           'religious', 'satellite',  'aid',  'missile',  'immigration',
           'synfuels', 'education',  'superfund', 'crime',
           'duty_free_exports',  'eaa_rsa']

all_header = party + headers

df = pd.read_csv(
    current_file + '\\house-votes-84.csv', names=all_header, na_values='?')
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
df[headers] = df[headers].apply(
    pd.to_numeric, errors='ignore', downcast='integer')
print(df.head)


# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
X_new = pd.DataFrame(np.array([[0.862783, 0.666076,  0.161553,
                                0.826788,  0.811522, 0.54612, 0.299998,
                                0.544518,  0.476537,  0.25485,  0.489953,
                                0.386101, 0.903861, 0.274487, 0.282128,
                                0.31358]]), columns=np.arange(0, 16))
print(X_new.info())
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
