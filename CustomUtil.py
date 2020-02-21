# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:21:05 2020

@author: arnaud
"""

''' Creation de fonctions utilitaires ''' 

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
from Preprocessing import *
from CNXManager import *
from CustomUtil import *
#https://spacy.io/usage
import spacy
nlp = spacy.load('fr_core_news_sm')


from nltk.corpus import stopwords
list_stopwords = stopwords.words('french')
list_stopwords.extend(['rt','être','avoir','voter','faire'])


def termFrequencyAll(tweets,tokenizer,list_stopwords):
    countTerm = Counter()
    for tweet in tweets:
        terms = [t for t in tokenizer(tweet) if t not in list_stopwords]
        countTerm.update(terms)
    return countTerm

''' 
Renvoie l histogramme des features les plus souvent rencontrees
counter : Counter() des features
maxNbFeatures : Nombre maximales de features à selectionner
'''
def plotHistogramMostCommonFeatures(counter,maxNbFeatures):
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
    bar_chart = plt.bar(x_pos,nb,width=0.5,align='center')
    plt.xticks(x_pos,features,rotation='vertical')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.yticks(y_pos)
    plt.xlabel('Nombre de fois que la feature a ete ecrite')
    plt.savefig('./Images/ffg')
    plt.show()
    