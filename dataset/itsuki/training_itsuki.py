import nltk

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

import json
with open("itsuki.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['itsuki']:
    for pola in intent['patterns']:
        kata = nltk.word_tokenize(pola)
        words.extend(kata)
        docs_x.append(kata)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
