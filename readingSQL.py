# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:02:24 2020

@author: arnau
"""

import numpy as np
import pandas as pd
import mysql.connector
import sklearn as sk
import nltk
import re

cnx = mysql.connector.connect(user='root', password='pldac',
                              host='127.0.0.1',
                              database='pldac')
cursor = cnx.cursor()
cursor.execute("select text from tweets_0415_0423 LIMIT 1")

tweets = list(cursor)
tweets = ''.join([tweets[i][0] for i in range(len(tweets))])
print(tweets)

stemmer = nltk.stem.snowball.FrenchStemmer()

a = "fatiguée"
a = stemmer.stem(a)
print(a)



cursor.close()
cnx.close()

