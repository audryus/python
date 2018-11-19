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


def answer(message):
    start = timeit.default_timer()
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
    print(_a)
    engine.say(_a)
    engine.runAndWait()
    return _a


for f in files:
    tree = ET.parse(f)
    root = tree.getroot()
    for cat in root.findall('category'):
        _p = cat.find('pattern').text
        questions[_p] = nlp(_p)


kernel = aiml.Kernel()
kernel.setBotPredicate("name", "Chief")

kernel.learn(current_file + "/startup.xml")
kernel.respond("load aiml")

while True:
    answer(input("> "))
