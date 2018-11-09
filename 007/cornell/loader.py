import sqlite3
import os
import re
import spacy

current_file = os.path.abspath(os.path.dirname(__file__))
connection = sqlite3.connect('conversation.db')
c = connection.cursor()

c.execute(""" CREATE TABLE IF NOT EXISTS conversation (
    input TEXT, output TEXT)""")

texts = {}
p = re.compile('L\d{1,}')
nlp = spacy.load('en')

inputs = []
outputs = []


def insert_data(data):
    sql = """INSERT INTO conversation(input, output) VALUES (?, ?)"""
    c = connection.cursor()
    c.execute(sql, data)


def clean_data(text):
    text = text.replace('\n', '')
    text = text.replace('...', '')
    text = text.replace('-', '').replace('--', '')
    text = text.replace("\'", "'")
    text = text.replace("*", "")
    return text


with open(current_file + '/mlines.txt') as f:
    for row in f:
        read = row.split('+++$+++')
        line = read[0]
        movie = read[2]
        text = read[4]
        key = movie.strip() + line.strip()
        texts[key] = clean_data(text)

missing = set()

with open(current_file + '/mconv.txt') as f:
    for row in f:
        read = row.split('+++$+++')
        movie = read[2]
        text = read[3]
        lines = p.findall(text)
        if len(lines) <= 1:
            continue
        for i in range(len(lines) - 1):
            key1 = movie.strip() + lines[i].strip()
            key2 = movie.strip() + lines[i + 1].strip()
            try:
                inputs.append(texts[key1])
                outputs.append(texts[key2])
            except KeyError:
                missing.add(lines[i].strip())
                missing.add(lines[i+1].strip())

for i in range(len(inputs)):
    insert_data((inputs[i], outputs[i]))

connection.commit()
