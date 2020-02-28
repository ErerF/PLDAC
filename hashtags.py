import pandas as pd
from sqlalchemy import create_engine
from Preprocessing import *
from CNXManager import *
from sqlalchemy import *
from CustomUtil import *
import time

pd.options.display.max_colwidth = 1000

''' Frequence des hashtags qui apparaissent au moins dix fois'''
engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")

pp = Preprocesser1()

vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',strip_accents='unicode',min_df=1000)

start = time.time()


tweets = pd.read_sql_query(
    "select text from tweets_0415_0423 where lang ='fr'", engine)
hashtags = pp.preprocessing(tweets['text'])
countMatrix = vectorizer.fit_transform(hashtags)
hashtags = vectorizer.get_feature_names()
print(len(hashtags))
frequency = np.asarray(countMatrix.sum(axis=0))
print(frequency.shape)
frequency = frequency.reshape((frequency.shape[1]))
print(frequency.shape)


df = pd.DataFrame(data= {'hashtags':hashtags,'frequency':frequency})


print("Temps pris en secondes : " + str(time.time() - start))
df.to_sql("dict_hashtags", engine, "pldac", if_exists='replace')