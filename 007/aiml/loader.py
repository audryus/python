import aiml
import os
import glob
import re
import spacy
import pandas as pd
import timeit
import random
import pyttsx3
from itertools import islice

start = timeit.default_timer()
# Load the spacy model: nlp
print('Starting ...')
nLines = 10000
nlp = spacy.load('en_core_web_lg')
engine = pyttsx3.init()
current_file = os.path.abspath(os.path.dirname(__file__))
questions = []
files = glob.glob(current_file + '/*.aiml')
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


def should_comment(message):
    if is_question(message):
        return True
    if ('hi' in message or 'hello' in message or 'hey' in message):
        return True
    return random.random() > 0.5


def is_question(message):
    is_question = False
    if (message.startswith('are ')
        or message.startswith('am I ')
        or message.startswith("ain't ")
        or message.startswith('is ')
        or message.startswith('what ')
            or message.startswith('who ')
            or message.startswith('how ')
        or message.startswith('whom ')
        or message.startswith('when ')
            or message.startswith('where ')
            or '?' in message):
        is_question = True
    return is_question


def answer(message):
    _s = timeit.default_timer()
    print('Looking for answer ...')
    if not should_comment(message):
        return ""
    message = message.lower()
    resp = nlp(message)
    _a = "Sorry. I don't understand."
    df['NLP_Result'] = df['Pattern_nlp'].apply(lambda l: l.similarity(resp))
    # Get the row highest values, to give a chance to be wrong
    nlargest = df.nlargest(2, 'NLP_Result')
    _q = nlargest.sample(1)['Pattern'].values[0]
    _a = kernel.respond(_q)
    print('Done answer:', (timeit.default_timer() - _s))
    return _a


start = timeit.default_timer()
print('Loading AIML ...')
kernel = aiml.Kernel()
kernel.setBotPredicate("name", "Chief")

kernel.learn(current_file + "/startup.xml")
kernel.respond("load aiml")
print('Done AIML: ', (timeit.default_timer() - start))


while True:
    _i = input("> ")
    _a = answer(_i)
    print(_a)
    engine.say(_a)
    engine.runAndWait()
