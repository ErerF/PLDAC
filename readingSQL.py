# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:02:24 2020

@author: arnau
"""

import numpy as np
import pandas as pd
import mysql.connector
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from collections import Counter
import json
import string

'''
Prends un tweet en entree de chaque fonction et le renvoie transforme

Faire une fonction qui correspond au traitement qu on veut sur les tweets

'''
def lowerCase(tweet):
    return tweet.lower()

def removeLinks(tweet):
    tweet = re.sub(r'https://[A-Za-z0-9_/.]+', '', tweet)
    return tweet

#rajout de l espace car certains tweets ne mettent pas d espace entre leurs mots et ponctuations
#on garde les hashtags 
def removePunctuation(tweet):
    tweet = "".join([c if c not in string.punctuation or c == '#' else " " for c in tweet ])
    return tweet

# verifier quoi utiliser pour le stemming
def stem(tweet):
    stemmer = nltk.stem.snowball.FrenchStemmer()
    tweet = " ".join(stemmer.stem(word) for word in tweet.split(" "))
    return stemmer.stem(tweet)

def lemmatise(tweet):
    return tweet

def removeMentions(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    return tweet

def removeNumbers(tweet):
    tweet = re.sub(r'[0-9]+','',tweet)
    return tweet



def removeSparseTerms(dicti,nbTerms):
    return 

cnx = mysql.connector.connect(user='root', password='pldac',
                              host='127.0.0.1',
                              database='pldac')
cursor = cnx.cursor()
cursor.execute("select text from tweets_0415_0423 LIMIT 3")
rows = cursor.fetchall()
tweets = pd.DataFrame(rows, columns=cursor.column_names)
pd.options.display.max_colwidth = 1000

print(tweets)
tweets['text'] = tweets['text'].apply(removeLinks)
tweets['text'] = tweets['text'].apply(removeMentions)
tweets['text'] = tweets['text'].apply(removePunctuation)
tweets['text'] = tweets['text'].apply(removeNumbers)
tweets['text'] = tweets['text'].apply(lowerCase)
tweets['text'] = tweets['text'].apply(stem)

print(tweets)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets['text'])
print(vectorizer.get_feature_names())
print(X.toarray())

counter = Counter(tweets['text'])
print()

cursor.close()
cnx.close()

