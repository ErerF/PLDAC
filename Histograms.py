
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np

from sqlalchemy import create_engine

engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")
hashtags = pd.read_sql_query(
    "select entropie1hashtags from entropie_tweets2 where frequence1 != ''", engine)
entropie = hashtags['entropie1hashtags'].values
temp = []

entropie = np.array([float(i) for i in entropie])

nb= 50
x = np.linspace(0,5,nb,dtype=float)
y = []
print(len(np.where(entropie >= 0)[0]))
for i in range(nb-1):
    y.append(len(np.where(np.logical_and(entropie >= x[i] , entropie <= x[i+1]))[0]))
print(x)
print(y)
plt.figure(figsize=(20,10))
plt.hist(entropie,x)
plt.title("entropie des utilisateurs")
plt.ylabel("Nombre d'utilisateurs")
plt.xlabel("Entropie")
#plt.show()
plt.savefig('./Images/entropieUtilisateurs2')

hashtags = pd.read_sql_query(
    "select hashtags,frequency from dict_hashtags", engine)
x = hashtags['hashtags'].values
y = hashtags['frequency'].values
y = [int(i) for i in y]
y = np.array(y)
idx = np.argsort(y)[::-1]
x = x[idx][:50]
y = y[idx][:50]
plt.bar(x,y)
plt.xticks(rotation=90)
plt.title("50 hashtags les plus utilisés")
plt.xlabel("Hashtags")
plt.ylabel("Nombre de fois utilisées")
#plt.show()
plt.savefig('./Images/topHashtags')




journalistes = pd.read_sql_query(
    "select entropie1hashtags from entropie_tweets2 e,users_0415_0423 u where e.user_id = u.user_id and e.frequence1 != '' and u.description like '%journalist%' ", engine)
entropie = journalistes['entropie1hashtags'].values
temp = []

entropie = np.array([float(i) for i in entropie])

nb= 50
x = np.linspace(0,5,nb,dtype=float)
y = []
print(len(np.where(entropie >= 0)[0]))
for i in range(nb-1):
    y.append(len(np.where(np.logical_and(entropie >= x[i] , entropie <= x[i+1]))[0]))
print(x)
print(y)
plt.figure(figsize=(20,10))
plt.hist(entropie,x)
plt.title("entropie des journalistes")
plt.ylabel("Nombre de journalistes")
plt.xlabel("Entropie")
#plt.show()
plt.savefig('./Images/entropieJournaliste')


journalistes = pd.read_sql_query(
    "select entropie1hashtags,e.followers_count from entropie_tweets2 e,users_0415_0423 u where e.user_id = u.user_id and e.frequence1 != '' and u.description like '%journalist%' ", engine)
entropie = journalistes['entropie1hashtags'].values
followers = journalistes['followers_count'].values
temp = []

entropie = np.array([float(i) for i in entropie])
followers = np.array([int(i) for i in followers])
nb= 50
x = np.linspace(0,5,nb,dtype=float)
y = []
print(len(np.where(entropie >= 0)[0]))
for i in range(nb-1):
    y.append(len(np.where(np.logical_and(entropie >= x[i] , entropie <= x[i+1]))[0]))
print(x)
print(y)
plt.figure(figsize=(20,10))
plt.scatter(entropie,followers)
plt.title("followers selon l'entropie")
plt.ylabel("Nombre de followers")
plt.xlabel("Entropie")
plt.yscale("log")
plt.ylim(bottom=1)
#plt.show()
plt.savefig('./Images/entropieFollowersJourna')


journalistes = pd.read_sql_query(
    "select entropie1hashtags,followers_count from entropie_tweets2 where frequence1 != '' ", engine)
entropie = journalistes['entropie1hashtags'].values
followers = journalistes['followers_count'].values
temp = []

entropie = np.array([float(i) for i in entropie])
followers = np.array([int(i) for i in followers])
nb= 50
x = np.linspace(0,5,nb,dtype=float)
y = []
print(len(np.where(entropie >= 0)[0]))
for i in range(nb-1):
    y.append(len(np.where(np.logical_and(entropie >= x[i] , entropie <= x[i+1]))[0]))
print(x)
print(y)
plt.figure(figsize=(20,10))
plt.scatter(entropie,followers)
plt.title("followers selon l'entropie")
plt.ylabel("Nombre de followers")
plt.xlabel("Entropie")
plt.yscale("log")
plt.ylim(bottom=1)
#plt.show()
plt.savefig('./Images/entropieFollowers')