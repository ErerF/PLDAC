import pandas as pd
from sqlalchemy import create_engine
from Preprocessing import *
from CNXManager import *
from sqlalchemy import *
from CustomUtil import *
import time

pd.options.display.max_colwidth = 1000

''' Frequence des cooccurences d'hashtags qui apparaissent au moins dix fois'''
engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")

offset_limit = 10000
pp = Preprocesser1()


start = time.time()
hashtags = pd.read_sql_query(
    "select hashtags,frequency from dict_hashtags ", engine)

vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',vocabulary=hashtags['hashtags'].values)
coOccurencesMatrix = np.zeros([hashtags.shape[0],hashtags.shape[0]])

hashtags = vectorizer.get_feature_names()
size = len(hashtags)

tweets = pd.read_sql_query(
    "select text from tweets_0415_0423 where lang ='fr'", engine)

corpus = pp.preprocessing(tweets['text'])
countMatrix = vectorizer.fit_transform(corpus)

print(countMatrix.shape)
frequency = np.asarray(countMatrix.sum(axis=0))
frequency = frequency.reshape((frequency.shape[1]))

coOccurencesMatrix = (countMatrix.T * countMatrix)
coOccurencesMatrix.setdiag(0)
coOccurencesMatrix = coOccurencesMatrix.toarray()

hashtags = vectorizer.get_feature_names()
hashtags = np.array(hashtags)

simi = {'hashtag':[],'top_ten':[],'top_ten_occ':[]}
'''
Prends les dix co occurences les plus utilisés pour chaque hashtag'''
print(size)
for i in range(size):
    row = coOccurencesMatrix[i]
    top_ten_idx = np.argpartition(row,-5)[-5:]
    top_ten_nb = np.partition(row, -5)[-5:]
    top_ten_coOcc = hashtags[top_ten_idx]
    simi['hashtag'].append(str(hashtags[i]))
    simi['top_ten'].append(str(top_ten_coOcc))
    simi['top_ten_occ'].append(str(top_ten_nb))

print(simi)
similarite = pd.DataFrame.from_dict(simi)
similarite.to_sql("dict_hashtags_cooccurences", engine, "pldac", if_exists='replace')

print("Temps pris en secondes : " + str(time.time() - start))
start = time.time()

''' Utilisation de la matrice de co occurences pour calculer la matrice de similarite de jaccard'''
import copy
jaccardMatrix = np.zeros([hashtags.shape[0],hashtags.shape[0]])
simi = {'hashtag':[],'top_ten':[],'top_ten_occ':[],'top_ten_coocc':[]}
for i in range(size):
    for j in range(size):
        f1 = frequency[i]
        f2 = frequency[j]
        intersec = coOccurencesMatrix[i][j]
        jaccardMatrix[i][j] = intersec/(f1 + f2 - intersec)


for i in range(size):
    row = jaccardMatrix[i]
    top_ten_idx = np.argpartition(row, -5)[-5:]
    top_ten_nb = np.partition(row, -5)[-5:]
    top_ten_coOcc = hashtags[top_ten_idx]
    simi['hashtag'].append(str(hashtags[i]))
    simi['top_ten'].append(str(top_ten_coOcc))
    simi['top_ten_occ'].append(str(top_ten_nb))
    simi['top_ten_coocc'].append(str(coOccurencesMatrix[i][top_ten_idx]))


similarite = pd.DataFrame(simi)
similarite.to_sql("dict_hashtags_jaccardcoef", engine, "pldac", if_exists='replace')
print("Temps pris en secondes : " + str(time.time() - start))
