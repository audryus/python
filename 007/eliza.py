import re
import os
import sqlite3
import nltk
from nltk.draw.tree import TreeView


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

print(sentence)
print(answers[1])

# tokens = nltk.word_tokenize(sentence)
tokens = nltk.sent_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
TreeView(entities)._cframe.print_to_file(current_file + '/output.ps')
