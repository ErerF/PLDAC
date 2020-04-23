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


pp = Preprocesser1()



start = time.time()
hashtags = pd.read_sql_query(
    "select hashtags,frequency from dict_hashtags where frequency > 2000", engine)

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
nbClusters = 40
model = SpectralCoclustering(n_clusters=nbClusters,random_state=1)
model.fit(coOccurencesMatrix)
print("fit")
print(coOccurencesMatrix)
fit_data = coOccurencesMatrix[np.argsort(model.row_labels_)]
fit_data= fit_data[:,np.argsort(model.column_labels_)]
hashtagsrow = hashtags[np.argsort(model.row_labels_)]
hashtagscolumns = hashtags[np.argsort(model.column_labels_)]
print("rowlavels")
print(model.row_labels_)
print("columnlzbels")
print(model.column_labels_)
print("hashtags")
print(hashtags)
print(fit_data.shape)
print(fit_data)
coOccurencesMatrix = fit_data

clusters = {}

for i in range(nbClusters):
    clusters[str(i)] = []

for a,b,c in zip(model.row_labels_,model.column_labels_,hashtags):
    print(c + " : " + str(a) + " " + str(b))
    clusters[str(a)].append(c)

for i in range(nbClusters):
    print(clusters[str(i)])
    print()

print("clusters")
clusters = list(clusters.items())
for i in range(len(clusters)):
    clusters[i] = str(' '.join(clusters[i][1]))
print(clusters)
hashtagsClusters = pd.DataFrame(clusters)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(50,50))

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
ax.set_xticklabels(hashtagscolumns)
ax.set_yticklabels(hashtagsrow)
ax.tick_params(labelsize=16)
plt.savefig('./Images/cooccurrencesMatrix2000_20.png')

fig = plt.figure(figsize=(50,100))
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

ax.tick_params(labelsize=40)


plt.savefig('./Images/cooccurrencesMatrix2.png')



plt.show()


sns.heatmap(coOccurencesMatrix,xticklabels=hashtags,yticklabels=hashtags)
sns.heatmap(coOccurencesMatrix2,xticklabels=hashtags,yticklabels=hashtags)



print("Temps pris en secondes : " + str(time.time() - start))




hashtagsClusters.to_sql ("clusters", engine, "pldac", if_exists='replace')