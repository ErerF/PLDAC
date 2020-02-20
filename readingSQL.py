# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:02:24 2020

"""

import numpy as np
import pandas as pd
import mysql.connector
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
list_stopwords.extend(['rt','être','avoir','voter','faire'])
'''
Prends un tweet en entree de chaque fonction et le renvoie transforme

Faire une fonction qui correspond au traitement qu on veut sur les tweets

'''

''' Debut methode de preprocessing '''
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
    if len(tweet) > 0:
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
    return ' '.join(hashtags)

def stripAccents(tweet):
    return tweet
''' fin methode preprocessing'''

''' Quelle est la regle statistique pour eliminer les tweets etrangers deja ? '''
def isFrench(tweet):
    if re.match('csgo',tweet) or re.match('giveaway',tweet):
        return false
    return true


''' Inutile deja dans count Vectorizer => max_features
def addStopWords(stopwordsInit,counter,nbMax):
    listWords = counter.most_common()[nbMax:]
    for w,f in listWords:
        stopwordsInit.append(w)
    return stopwordsInit
'''

def termFrequencyAll(tweets,tokenizer,list_stopwords):
    countTerm = Counter()
    for tweet in tweets:
        terms = [t for t in tokenizer(tweet) if t not in list_stopwords]
        countTerm.update(terms)
    return countTerm

def plotHistogram(counter,maxNbFeatures):
    listTermsFreq = counter.most_common(maxNbFeatures)
    features= list()
    nb = list()
    for k,v in listTermsFreq:
        features.append(k)
        nb.append(v)
    
    y_pos = np.linspace(0,max(nb)+1,10,dtype = int)
    print(y_pos)
    x_pos = np.linspace(0,len(features),maxNbFeatures)
    print(x_pos)
    plt.bar(x_pos,nb,width=0.5,align='center')
    plt.xticks(x_pos,features)
    plt.yticks(y_pos)
    plt.xlabel('Nombre de fois que la feature a ete ecrite')
    plt.show()
    
    
    
f=open("database_info.json")
db_info=json.load(f)
u=db_info["user"]
pwd=db_info["pwd"]
h=db_info["host"]
db=db_info["database"] 
    
cnx = mysql.connector.connect(user=u, password=pwd,
                              host=h,
                              database=db)
cursor = cnx.cursor()
cursor.execute("select text from tweets_0415_0423 where lang ='fr' LIMIT 1000000")
rows = cursor.fetchall()
tweets = pd.DataFrame(rows, columns=cursor.column_names)
pd.options.display.max_colwidth = 1000

print(tweets)
#tweets['text'] = tweets['text'].apply(removeLinks)
#tweets['text'] = tweets['text'].apply(removeMentions)
#tweets['text'] = tweets['text'].apply(removePunctuation)
#tweets['text'] = tweets['text'].apply(removeNumbers)
#tweets['text'] = tweets['text'].apply(lowerCase)
#tweets['text'] = tweets['text'].apply(stem)
tweets['text'] = tweets['text'].apply(listHashtags)
print("lemmatize")
#tweets['text'] = tweets['text'].apply(lemmatize)

        

print(tweets)
''' (?u) = re.UNICODE ,\b = delimiteur d un mot
la regex permet de garder aussi les hashtags dans le passage dans countVectorizer

'''
maxFeatures = 15
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',stop_words = list_stopwords,max_features = maxFeatures )

tokenizer = vectorizer.build_analyzer()


'''
Histogramme des mots cles (les plus souvents utilisés)
Addition des lignes de la matrice 
ordre decroissant
garder que les n mots les plus utilises

Trop lent :
print("NpSum")
features = np.array(vectorizer.get_feature_names())

sum_rows = np.sum(X,axis=0)
sum_rows = np.array(sum_rows)
sum_rows = sum_rows.reshape(-1)
sortOrder = (-sum_rows).argsort()

nb = sum_rows[sortOrder]
features = features[sortOrder]
'''

print("Histogramme")
counter = termFrequencyAll(tweets['text'],tokenizer,list_stopwords)
plotHistogram(counter,maxFeatures )


#list_stopwords = addStopWords(list_stopwords, counter ,maxNbFeatures)

'''
Classifier tweets selon mots clés
Ajouter à la liste des stop words, les mots considérés comme non importants
'''
countMatrix = vectorizer.fit_transform(tweets['text'])
print(vectorizer.get_feature_names())
print(countMatrix.toarray())

tfidfMatrix = TfidfTransformer().fit_transform(countMatrix)
print(tfidfMatrix.toarray())





print(list_stopwords)


cursor.close()
cnx.close()

