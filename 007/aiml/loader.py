import aiml
import os
import glob
import re
import spacy
import pandas as pd
import timeit
import random
from itertools import islice

start = timeit.default_timer()
# Load the spacy model: nlp
print('Starting ...')
nLines = 10000
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg')
current_file = os.path.abspath(os.path.dirname(__file__))
questions = []
files = glob.glob(current_file + '/bot.aiml')
p = re.compile("<pattern>(.*)</pattern>")
df = None
print('Done start: ', (timeit.default_timer() - start))

start = timeit.default_timer()
print('Reading Files ...')
for f in files:
    with open(f) as _f:
        while True:
            next_n_lines = list(islice(_f, nLines))
            if not next_n_lines:
                break
            for row in next_n_lines:
                lines = p.findall(row)
                if len(lines) == 0:
                    continue
                _p = lines[0]
                questions.append(_p)
df = pd.DataFrame(questions, columns=['Pattern'])
questions = None
print('Done Files: ', (timeit.default_timer() - start))

start = timeit.default_timer()
print('NLP Patterns')
df["Pattern_nlp"] = df["Pattern"].apply(lambda l: nlp(l))
print('Done NLP: ', (timeit.default_timer() - start))

df.to_pickle(current_file + '/alice.pkl', compression='gzip')
