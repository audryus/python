import xml.etree.ElementTree as ET
import re
import os
import sqlite3
import nltk
import numpy as np
import spacy
import timeit
# Load the spacy model: nlp
nlp = spacy.load('en_core_web_lg')


def replace_pronouns(message):
    message = message.lower()
    if 'me' in message:
        # Replace 'me' with 'you'
        return re.sub('me', 'you', message)
    if 'my' in message:
        # Replace 'my' with 'your'
        return re.sub('my', 'your', message)
    if 'your' in message:
        # Replace 'your' with 'my'
        return re.sub('your', 'my', message)
    if 'you' in message:
        # Replace 'you' with 'me'
        return re.sub('you', 'me', message)

    return message


current_file = os.path.abspath(os.path.dirname(__file__))

connection = sqlite3.connect('conversation.db')
c = connection.cursor()
query = "SELECT * FROM conversation"
result = c.execute(query).fetchall()

questions = []
answers = []

for row in result:
    if '?' in row[0]:
        questions.append(row[0])
        answers.append(row[1])

sentence = questions[1]
sentences = questions[:1]
print(sentence)
print(answers[1])

# tokens = nltk.word_tokenize(sentence)
tokens = nltk.sent_tokenize(sentence)
token = tokens[-1]
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)


def extract_entities(message):
    # Create a dict to hold the entities
    doc = nlp(message)
    for token in doc:
        print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
            token.text,
            token.idx,
            token.lemma_,
            token.is_punct,
            token.is_space,
            token.shape_,
            token.pos_,
            token.tag_
        ))


# print(extract_entities(sentence))
# print(extract_entities(answers[1]))


targets = []
qna = []
tree = ET.parse(current_file + '/aiml/bot.aiml')
root = tree.getroot()

for country in root.findall('category'):
    _p = country.find('pattern').text
    _t = country.find('template').text
    if _t:
        targets.append(nlp(_p))
        qna.append(_t)

print(len(targets))


def answer(message):
    start = timeit.default_timer()
    resp = nlp(message)
    _a = "Sorry. I don't understand."
    result = np.zeros(len(targets))
    for i, t in enumerate(targets):
        result[i] = t.similarity(resp)
    _ima = np.nanargmax(result)
    if len(qna) >= _ima:
        _a = qna[_ima]
    print('Time: ', timeit.default_timer() - start)
    return _a


while True:
    print(answer(input("> ")))
