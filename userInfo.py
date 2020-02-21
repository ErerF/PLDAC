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
from sqlalchemy import *
from CustomUtil import *
#https://spacy.io/usage
import spacy
nlp = spacy.load('fr_core_news_sm')


from nltk.corpus import stopwords
list_stopwords = stopwords.words('french')
list_stopwords.extend(['rt','être','avoir','voter','faire'])

''' definition des hashtags utilises pour chaque candidat'''
hamon_list = ["#futurdesirable"]
melenchon_list = ["#jlm2017","#laforcedupeuple","#insoumis","#franceinsoumise"]
lepen_list = ["#lepen","#mlp","#fn"]
macron_list = ["#teammacron","#jevotemacron","#macron2017","#enmarche","#ladroitevotemacron","#macronpresident"]
fillon_list = ["#fillon","#lr"]

''' Creation de la connection à la BD et recuperation des donnees '''
#password :
#pldac ou root
cnxman = CNXManager('root','root','pldac')
print("query selects users")
rows = cnxman.executeSelectQuery("select user_id,followers_count,friends_count from users_0415_0423 where lang ='fr' LIMIT 30 OFFSET 15  000")

users = pd.DataFrame(rows, columns=cnxman.cursor.column_names)
pd.options.display.max_colwidth = 1000
users['user_id'] = users['user_id'].apply(str)

usersIds = "' or user_id = '".join(users['user_id'].tolist())
print('query select tweets')
print("select text,user_id from tweets_0415_0423 where lang ='fr' and (user_id = ' " + usersIds + "')")
rows = cnxman.executeSelectQuery(
        "select text,user_id from tweets_0415_0423 where lang ='fr' ") #and (user_id = ' " + usersIds + "')")
tweets = pd.DataFrame(rows, columns=cnxman.cursor.column_names)
tweets['user_id'] = tweets['user_id'].apply(str)

''' label politiques des users selon leurs hashtags les plus utilises'''
label = {}

''' Selection pour chaque user de ses tweet et etudes de ses hashtags'''
pp = Preprocesser1()
maxFeatures = 10
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',max_features = maxFeatures )
vectorizer2 = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',max_features = maxFeatures, ngram_range=(2,2) )

''' Recuperer tous les tweets des users pris d'un coup puis faire le where dans panda dataframe'''
print("for sur les users")
users['hashtags'] = np.array(("empty"))
users['grpPolitique'] = np.array(("empty"))
users['hashtags'] = users['hashtags'].apply(str)
users['grpPolitique'] = users['grpPolitique'].apply(str)

for index,user in users.iterrows():
    userTweets = tweets.loc[tweets['user_id'] == user['user_id']]
    empty = False
    #print(userTweets['text'])
    try:
        userTweets['text'] = pp.preprocessing(userTweets['text'])
        countMatrix = vectorizer.fit(userTweets['text'])
        print(vectorizer.get_feature_names())
        hashtags = vectorizer.get_feature_names()
    except ValueError:
        empty = True
    if not empty:
        ''' Selection des hashtags les plus utilises par le user
        On determine son label politique selon ces hashtags et des listes predefinies
        Clustering qui permettrait de mieux faire cela ?'''
        #for hashtag in hashtags:
        users.at[index,'hashtags'] = ";".join(hashtags)
        hashtags_threshold = len(hashtags) / 3
        if len(set(melenchon_list) & set(hashtags)) >= hashtags_threshold:
            users.at[index,'grpPolitique'] = "FI"
        elif len(set(macron_list) & set(hashtags)) >= hashtags_threshold:
            users.at[index,'grpPolitique'] = "LREM"
        elif len(set(fillon_list) & set(hashtags)) >= hashtags_threshold:
            users.at[index,'grpPolitique'] = "LR"
        elif len(set(lepen_list) & set(hashtags)) >= hashtags_threshold:
            users.at[index,'grpPolitique'] = "FN"
        elif len(set(hamon_list) & set(hashtags)) >= hashtags_threshold:
            users.at[index,'grpPolitique'] = "PS"
        else:
            users.at[index,'grpPolitique'] = "UNK"
        print(users.at[index,'grpPolitique'])
        print(users.at[index,'hashtags'] )

cnxman.closeCNX()

print(users)

print("ecriture des resultats dans la base")
users.to_csv(path_or_buf="./CSV/userstest.csv",)

