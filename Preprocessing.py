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
        hashtags = re.findall(r'#\S+',tweet)
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

class Preprocesser1(preprocesserGeneral):
    
    def preprocessing(self,tweets):
        #tweets['text'] = tweets['text'].apply(removeLinks)
        #tweets['text'] = tweets['text'].apply(removeMentions)
        #tweets['text'] = tweets['text'].apply(removePunctuation)
        #tweets['text'] = tweets['text'].apply(removeNumbers)
        #tweets['text'] = tweets['text'].apply(lowerCase)
        #tweets['text'] = tweets['text'].apply(stem)
        tweets= tweets.apply(self.listHashtags)
        print("lemmatize")
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
              
