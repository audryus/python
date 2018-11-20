import aiml
import os
import pyttsx3
import spacy
import pandas as pd
import timeit
import random
import pickle

start = timeit.default_timer()
# Load the spacy model: nlp
print('Starting ...')
nLines = 10000
nlp = spacy.load('en_core_web_lg')
current_file = os.path.abspath(os.path.dirname(__file__))
engine = pyttsx3.init()
df = pd.read_pickle(current_file + '/alice.pkl')
print('Done start: ', (timeit.default_timer() - start))


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