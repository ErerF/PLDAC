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
    plt.bar(x_pos,nb,width=0.5,align='center')
    plt.xticks(x_pos,features)
    plt.yticks(y_pos)
    plt.xlabel('Nombre de fois que la feature a ete ecrite')
    plt.savefig('./Images/histogramMostCommon')
    plt.show()
    
    
''' Creation de la connection à la BD et recuperation des donnees '''
f=open("database_info.json")
db_info=json.load(f)
u=db_info["user"]
pwd=db_info["pwd"]
db=db_info["database"] 
cnxman = CNXManager(u,pwd,db)
rows = cnxman.executeSelectQuery("select text from tweets_0415_0423 where lang ='fr' LIMIT 1000000")

tweets = pd.DataFrame(rows, columns=cnxman.cursor.column_names)
pd.options.display.max_colwidth = 1000

print(tweets)

''' Choix du preprocesseur'''
pp = Preprocesser1()
tweets['text'] = pp.preprocessing(tweets['text'])

print(tweets)




''' Transformation du texte traite en ce que l on souhaite  et experimentation dessus'''

''' (?u) = re.UNICODE ,\b = delimiteur d un mot
la regex permet de garder aussi les hashtags dans le passage dans countVectorizer

'''


maxFeatures = 15
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',stop_words = list_stopwords,max_features = maxFeatures )
tokenizer = vectorizer.build_analyzer()



'''

Creation de graphiques du compte rendu des experimentations 


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
plotHistogramMostCommonFeatures(counter,maxFeatures )


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

cnxman.closeCNX()

