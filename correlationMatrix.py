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
print(coOccurencesMatrix)
print(coOccurencesMatrix.shape)

hashtags = vectorizer.get_feature_names()
hashtags = np.array(hashtags)

corrMatrix = np.corrcoef(coOccurencesMatrix)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)
from matplotlib.colors import LogNorm
cax = ax.matshow(coOccurencesMatrix,vmax=1500 )

fig.colorbar(cax)
plt.xticks(rotation=90)


ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticklabels(hashtags)
ax.set_yticklabels(hashtags)

plt.savefig('./Images/cooccurrencesMatrix.png')
plt.show()


print("Temps pris en secondes : " + str(time.time() - start))




