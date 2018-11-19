import aiml
import os
import glob
import re
import pyttsx3
import xml.etree.ElementTree as ET
import spacy
import numpy as np
import timeit
import random

# Load the spacy model: nlp
nlp = spacy.load('en_core_web_lg')
current_file = os.path.abspath(os.path.dirname(__file__))
engine = pyttsx3.init()

files = glob.glob(current_file + '/*.aiml')
# questions = []
# qna = []
questions = {}


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
        or message.startswith('whom ')
        or message.startswith('when ')
            or message.startswith('where ')
            or '?' in message):
        is_question = True
    return is_question


def answer(message):
    if not should_comment(message):
        return ""
    start = timeit.default_timer()
    message = message.lower()
    resp = nlp(message)
    _a = "Sorry. I don't understand."
    result = 0
    qList = set()
    for k, v in questions.items():
        _s = v.similarity(resp)
        if _s > result:
            result = _s
            qList.clear()
            qList.add(k)
        elif _s == result:
            qList.add(k)
    _q = random.choice(list(qList))
    _a = kernel.respond(questions[_q])
    return _a


for f in files:
    tree = ET.parse(f)
    root = tree.getroot()
    for cat in root.findall('category'):
        _p = cat.find('pattern').text
        if _p is None:
            print(f)
        elif len(_p.strip()) > 0 and _p is not "*":
            questions[_p] = nlp(_p)


kernel = aiml.Kernel()
kernel.setBotPredicate("name", "Chief")

kernel.learn(current_file + "/startup.xml")
kernel.respond("load aiml")

while True:
    _i = input("> ")
    _a = answer(_i)
    print(_a)
    engine.say(_a)
    engine.runAndWait()
