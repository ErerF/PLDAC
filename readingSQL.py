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
import matplotlib.pyplot as plt

#https://spacy.io/usage
import spacy
nlp = spacy.load('fr_core_news_sm')


from nltk.corpus import stopwords
list_stopwords = stopwords.words('french')

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

# a tester pour comparer avec le stem, il met hashtag dans un token propre
def lemmatize(tweet):
    doc = nlp(tweet)
    tweet =  " ".join(token.lemma_ for token in doc)
    return tweet
    

def removeMentions(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    return tweet

def removeNumbers(tweet):
    tweet = re.sub(r'[0-9]+','',tweet)
    return tweet


def listHashtags(tweet):
    hashtags = re.findall(r'#\S+',tweet)
    return hashtags

def removeNonFrench():
    return

def removeSparseTerms(dicti,nbTerms):
    return 

def plotHistogram(features,nb,maxNbFeatures):
    if len(features) > maxNbFeatures:
        features= features[:maxNbFeatures]
        nb = nb[:maxNbFeatures]
        
    y_pos = np.arange(max(nb)+1)
    x_pos = np.arange(len(features))
    plt.bar(x_pos,nb,width=0.5,align='center')
    plt.xticks(x_pos,features)
    plt.yticks(y_pos)
    plt.xlabel('Nombre de fois que la feature a ete ecrite')
    plt.show()
#password :
#pldac
#root
cnx = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='pldac')
cursor = cnx.cursor()
cursor.execute("select text from tweets_0415_0423 LIMIT 30")
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
#tweets['text'] = tweets['text'].apply(lemmatize)



print(tweets)
''' (?u) = re.UNICODE ,\b = delimiteur d un mot
la regex permet de garder aussi les hashtags dans le passage dans countVectorizer

'''
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',stop_words=list_stopwords)
X = vectorizer.fit_transform(tweets['text'])
print(vectorizer.get_feature_names())
print(X.toarray())
features = np.array(vectorizer.get_feature_names())

'''
Histogramme des mots cles (les plus souvents utilisés)
Addition des lignes de la matrice 
ordre decroissant
garder que les n mots les plus utilises
'''
print("NpSum")
sum_rows = np.sum(X,axis=0)
sum_rows = np.array(sum_rows)
sum_rows = sum_rows.reshape(-1)
sortOrder = (-sum_rows).argsort()

nb = sum_rows[sortOrder]
features = features[sortOrder]

print("Histogramme")
plotHistogram(features,nb,15  )

maxNbFeatures = 100
features= features[:maxNbFeatures]
nb = nb[:maxNbFeatures]
print(features)
print(nb)
'''
Classifier tweets selon mots clés
'''



cursor.close()
cnx.close()

