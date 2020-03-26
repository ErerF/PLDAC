import pandas as pd
from sqlalchemy import create_engine
from Preprocessing import *
from CNXManager import *
from sqlalchemy import *
from CustomUtil import *
import time
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster.bicluster import SpectralBiclustering

from sklearn.preprocessing import StandardScaler


import seaborn as sns

pd.options.display.max_colwidth = 1000

''' Frequence des cooccurences d'hashtags qui apparaissent au moins dix fois'''
engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")

offset_limit = 10000
pp = Preprocesser1()



start = time.time()
hashtags = pd.read_sql_query(
    "select hashtags,frequency from dict_hashtags where frequency > 6000", engine)

vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',vocabulary=hashtags['hashtags'].values)
coOccurencesMatrix = np.zeros([hashtags.shape[0],hashtags.shape[0]])

hashtags = vectorizer.get_feature_names()
size = len(hashtags)

tweets = pd.read_sql_query(
    "select text from tweets_0415_0423 where lang ='fr' limit 10000000000 ", engine)

corpus = pp.preprocessing(tweets['text'])
countMatrix = vectorizer.fit_transform(corpus)



coOccurencesMatrix = (countMatrix.T * countMatrix)
print(type(coOccurencesMatrix))

coOccurencesMatrix = coOccurencesMatrix.toarray()
np.fill_diagonal(coOccurencesMatrix,0)
print(coOccurencesMatrix)
print(coOccurencesMatrix.shape)

hashtags = vectorizer.get_feature_names()
hashtags = np.array(hashtags)


coOccurencesMatrix = np.where(coOccurencesMatrix == 0, 0,coOccurencesMatrix)
#coOccurencesMatrix = StandardScaler().fit_transform(coOccurencesMatrix)

print(coOccurencesMatrix)
import copy
coOccurencesMatrix2 = copy.deepcopy(coOccurencesMatrix)
coOccurencesMatrix2 = np.corrcoef(coOccurencesMatrix2)
coOccurencesMatrix = np.corrcoef(coOccurencesMatrix)
model = SpectralCoclustering(n_clusters=5,random_state=1)
model.fit(coOccurencesMatrix)
print("fit")
print(coOccurencesMatrix)
fit_data = coOccurencesMatrix[np.argsort(model.row_labels_)]
fit_data= fit_data[:,np.argsort(model.column_labels_)]
print("rowlavels")
print(model.row_labels_)
print("columnlzbels")
print(model.column_labels_)
print("hashtags")
print(hashtags)
print(fit_data.shape)
print(fit_data)
coOccurencesMatrix = fit_data

clusters = {"0":[],"1":[],"2":[],"3":[],"4":[]}

for a,b,c in zip(model.row_labels_,model.column_labels_,hashtags):
    print(c + " : " + str(a) + " " + str(b))
    clusters[str(a)].append(c)

print(clusters)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
from matplotlib.colors import LogNorm
#print("cooccr1 : reordered")
plt.title(" reordered")
#print(coOccurencesMatrix)
cax = ax.matshow(coOccurencesMatrix)

fig.colorbar(cax)
plt.xticks(rotation=90)


ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticklabels(hashtags)
ax.set_yticklabels(hashtags)

plt.savefig('./Images/cooccurrencesMatrix.png')

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.title(" non reordered")
from matplotlib.colors import LogNorm
#print("cooccurence2 : non reordered")
#print(coOccurencesMatrix2)
cax = ax.matshow(coOccurencesMatrix2)

fig.colorbar(cax)
plt.xticks(rotation=90)


ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticklabels(hashtags)
ax.set_yticklabels(hashtags)
plt.savefig('./Images/cooccurrencesMatrix2.png')



plt.show()


sns.heatmap(coOccurencesMatrix,xticklabels=hashtags,yticklabels=hashtags)
sns.heatmap(coOccurencesMatrix2,xticklabels=hashtags,yticklabels=hashtags)



print("Temps pris en secondes : " + str(time.time() - start))




