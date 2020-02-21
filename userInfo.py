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

''' definition des hashtags utilises pour chaque candidat'''
hamon_list = []
melenchon = []
lepen_list = []
macron_list = []
fillon_list = []

''' Creation de la connection à la BD et recuperation des donnees '''
#password :
#pldac ou root
cnxman = CNXManager('root','root','pldac')
rows = cnxman.executeSelectQuery("select user_id,description,followers_count,friends_count from users_0415_0423 where lang ='fr' LIMIT 100")

users = pd.DataFrame(rows, columns=cnxman.cursor.column_names)
pd.options.display.max_colwidth = 1000
users['user_id'] = users['user_id'].apply(str)
usersIds = "' or user_id = '".join(users['user_id'].tolist())
rows = cnxman.executeSelectQuery(
        "select text,user_id from tweets_0415_0423 where lang ='fr' and (user_id = ' " + usersIds + "')")
tweets = pd.DataFrame(rows, columns=cnxman.cursor.column_names)

''' label politiques des users selon leurs hashtags les plus utilises'''
label = np.zeros(len(users.index))
''' Selection pour chaque user de ses tweet et etudes de ses hashtags'''
pp = Preprocesser1()
maxFeatures = 3
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',max_features = maxFeatures )
tokenizer = vectorizer.build_analyzer()

''' Recuperer tous les tweets des users pris d'un coup puis faire le where dans panda dataframe'''

for index,user in users.iterrows():
    userTweets = tweets.loc[tweets['user_id'] == user['user_id']]
    print(userTweets)
    userTweets['text'] = pp.preprocessing(userTweets['text'])
    countMatrix = vectorizer.fit_transform(userTweets['text'])
    print(vectorizer.get_feature_names())


cnxman.closeCNX()

