# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:38:07 2020

@author: arnaud
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
macron = set(("#macron","#enmarche","#macronbercy","#jevotemacron","#macronpresident","#bercy","#teammacron"))
melenchon = set(("#jevotemelenchon","#lafranceinsoumise","#aperoinsoumis","#aperosinsoumis","#insoumis","#melenchon","#erepublique","#avenirencommun","#jlm","#jl","#jlmdijon","#jlmelenchon","#jlmhologramme","#keepcalmandvotemelenchon","#laforcedupeuple","#penicheinsoumise","#feuillederoutefi"))
hamon = set(("#benoithamon","#jevotehamon","#hamon","#coeurbattant","#hamontour","#futurdesirable","#fairebattrelecoeurdelafrance","#revenuuniversel"))
lepen = set(("#fn","#mlp","#marine","#jevotemarine","#marineaparis","#aunomdupeuple","#marinepournosanimaux","#marinepresidente","#marseillemlp","#perpignanmlp","#avecmarine","#lasecuritecestmarine"))
fillon = set(("#jevotefillon","fillonlille","#fillonmontpellier","#fillonpresident","#fillonnice","#fillonpresident","#fillon","#jevotefillondeslepremiertour","#senscommun","#lr"))
'''

macron = set(("#macron","#macrno","#macro","#macrob","#macrin","#"))
hamon = set(("#hammon","#hammond","#hamn","#hamo","#hamo","#hamob","#hamobpresident","#hamom","#hamon","#hamonb","#hamono","#hamonpresident"))
melenchon = set(("#melenchon","#melanchon","#melechon","#mélenchon","#mélanchon","#melechon","#jlm","#jlcm","#jlmo","#jlmelenchon"))
lepen = set(("#marine","#marin","#lepen","#lepe","#lepan","#lepem"))
fillon = set(("#fillon","#filln","#fillo","#fillob","#fillom","#fillojn"))

''' 
Definition des objects permettant de preprocesser les tweets
Un preprocesseur prend en entrée une colonne d'un dataFrame pandas de tweets 
et la renvoie traité

Dans la classe mere : toutes les methodes de transformation d'un tweet
Dans les classes filles : choix des methodes de traitement pour le corpus

Il faut définir un preprocesseur en implementant la fonction preprocessing puis 
lui donner la colonne des tweets d'un dataFrame
'''

class preprocesserGeneral():
    
    def preprocessing(self,tweets):
        raise NotImplementedError()
           
        '''
    Prends un tweet en entree de chaque fonction et le renvoie transforme
    
    Faire une fonction qui correspond au traitement qu on veut sur les tweets
    
    '''
    
    ''' Debut methode de preprocessing '''
    def lowerCase(self,tweet):
        return tweet.lower()
    
    def removeLinks(self,tweet):
        tweet = re.sub(r'https://[A-Za-z0-9_/.]+', '', tweet)
        return tweet
    
    #rajout de l espace car certains tweets ne mettent pas d espace entre leurs mots et ponctuations
    #on garde les hashtags 
    def removePunctuation(self,tweet):
        tweet = "".join([c if c not in string.punctuation or c == '#' else " " for c in tweet ])
        return tweet
    
    # verifier quoi utiliser pour le stemming
    def stem(self,tweet):
        stemmer = nltk.stem.snowball.FrenchStemmer()
        tweet = " ".join(stemmer.stem(word) for word in tweet.split(" "))
        return stemmer.stem(tweet)
    
    # a tester pour comparer avec le stem, il met hashtag dans un token propre
    def lemmatize(self,tweet):
        if len(tweet) > 0:
            doc = nlp(tweet)
            tweet =  " ".join(token.lemma_ for token in doc)
        return tweet
        
    
    def removeMentions(self,tweet):
        tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
        return tweet
    
    def removeNumbers(self,tweet):
        tweet = re.sub(r'[0-9]+','',tweet)
        return tweet
    
    
    def listHashtags(self,tweet):
        hashtag_re = re.compile("(?:^|\s)([#]{1}\w+)",re.UNICODE)
        hashtags = hashtag_re.findall(tweet)
        return ' '.join(hashtags)
    
    def stripAccents(self,tweet):
        return tweet
    ''' fin methode preprocessing'''
    
    ''' Quelle est la regle statistique pour eliminer les tweets etrangers deja ? '''
    def isFrench(self,tweet):
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



    def grpHashtag(self,tweet):
        h = tweet.split(" ")
        for i in range(len(h)):
            if h[i] in macron:
                h[i] = "#macron"
            elif h[i] in melenchon:
                h[i] = "#melenchon"
            elif h[i] in hamon:
                h[i] = "#hamon"
            elif h[i] in lepen:
                h[i] = "#lepen"
            elif h[i] in fillon:
                h[i] = "#fillon"
        tweet = " ".join(h)
        return tweet

class Preprocesser1(preprocesserGeneral):
    
    def preprocessing(self,tweets):
        #tweets['text'] = tweets['text'].apply(removeLinks)
        #tweets['text'] = tweets['text'].apply(removeMentions)
        #tweets['text'] = tweets['text'].apply(removePunctuation)
        tweets = tweets.apply(self.removeNumbers)
        tweets = tweets.apply(self.lowerCase)
        #tweets['text'] = tweets['text'].apply(stem)

        tweets = tweets.apply(self.grpHashtag)
        tweets= tweets.apply(self.listHashtags)
        #tweets['text'] = tweets['text'].apply(lemmatize))
       
        return tweets


class Preprocesser2(preprocesserGeneral):
    
    def preprocessing(self,tweets):
        tweets = tweets.apply(removeLinks)
        tweets = tweets.apply(removeMentions)
        tweets = tweets.apply(removePunctuation)
        tweets = tweets.apply(removeNumbers)
        tweets = tweets.apply(lowerCase)
        tweets = tweets.apply(stem)
        #tweets= tweets.apply(self.listHashtags)
        print("lemmatize")
        #tweets['text'] = tweets['text'].apply(lemmatize))
       
        return tweets
              
def preprocess(tweets,links = False,mentions = False,punctuation = False,numbers = False,lower = False,stemming = False,hashtags = False):
    if links:
        tweets = tweets.apply(removeLinks)
    if mentions:
        tweets = tweets.apply(removeMentions)
    if punctuation:
        tweets = tweets.apply(removePunctuation)
    if numbers:
        tweets = tweets.apply(removeNumbers)
    if lower:
        tweets = tweets.apply(lowerCase)
    if stemming:
        tweets = tweets.apply(stem)
    if hashtags:
        tweets = tweets.apply(listHashtags)
    return tweets