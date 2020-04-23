import pandas as pd
from sqlalchemy import create_engine
from Preprocessing import *
from CNXManager import *
from sqlalchemy import *
from CustomUtil import *
import time



pd.options.display.max_colwidth = 1000

''' Recuperation Users'''
engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")

#tweets = pd.read_sql_query(
#    "select user_id, text from tweets_0415_0423 where lang ='fr' ", engine)
print("fin query tweet")
start = time.time()

clusters = pd.read_sql_query(
    "select * from clusters", engine)
clusters = clusters.values.tolist()
clusters = [str(i[1] + " ")  for i in clusters]
print("clusters")
print(clusters)

users = pd.read_sql_query(
    "select user_id,followers_count,friends_count from users_0415_0423 where lang ='fr'  limit 150000  offset 30 0000 ", engine)



print("Fin des query")
''' Toujours faire de la vectorisation en préférant numpy à pandas'''
#tweets['user_id'] = tweets['user_id'].values.astype(str)
users['user_id'] = users['user_id'].values.astype(str)
users['nbTweets'] = 0
users['hashtags1gram'] = ""
users['frequence1'] = ""
users['entropie1hashtags'] = ""
users['hashtags2gram'] = ""
users['frequence2'] = ""
users['entropie2hashtags'] = ""

''' Selection pour chaque user de ses tweet et etudes de ses hashtags'''
pp = Preprocesser1()
maxFeatures = 50
vectorizer = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',max_features = maxFeatures)
vectorizer2 = CountVectorizer(token_pattern= r'(?u)#?\b\w\w+\b',max_features = maxFeatures, ngram_range=(2,2) )

'''Frequence hashtags'''
hashtagsFrequency = pd.read_sql_query(
    "select hashtags,frequency from dict_hashtags", engine)


print('Iteration sur les users')
indxDelete = []
cpt = 0
longueur = len(users )
for index,user in users.iterrows():
    #userTweets = tweets.loc[tweets['user_id'] == user['user_id']]
    cpt = cpt + 1
    if cpt % 100 == 0:
        print(str(cpt) + " sur " + str(longueur))
    userTweets = pd.read_sql_query(
        "select user_id, text from tweets_0415_0423 where user_id =  " +str(user['user_id']), engine)
    nbTweets = userTweets.shape[0]
    if nbTweets >= 100:
        try:
            users.at[index,'nbTweets'] = nbTweets
            hashtags = pp.preprocessing(userTweets['text'])
            countMatrix = vectorizer.fit_transform(hashtags)
            hashtags = vectorizer.get_feature_names()
            frequencyNonClusters = np.asarray(countMatrix.sum(axis=0))

            '''
            On met les hashtags dans leur cluster
            '''
            frequency = list(np.zeros(len(clusters),dtype=int))
            for i in range(len(hashtags)):
                for j in range(len(clusters)):
                    if str(hashtags[i] + " ") in clusters[j]:
                        frequency[j] = frequency[j] + frequencyNonClusters[0][i]

            frequency = np.array(frequency)
            if len(hashtags) > 0 and np.sum(frequencyNonClusters) >= 1:
                listt = [ str((h,f)) for h,f in zip(hashtags,frequencyNonClusters[0])]

                users.at[index, 'hashtags1gram'] = " ".join(listt)
                users.at[index, 'frequence1'] = " ".join(map(str, frequencyNonClusters))
                #print(users.at[index, 'hashtags1gram'] )
                total = frequency.sum()
                print(total)
                proba = frequency / total
                proba = np.where(proba == 0,0.000000000000000000000000001,proba)
                print(proba)
                entropy = - np.sum( proba * np.log2(proba))
                print(entropy)
                users.at[index, 'entropie1hashtags'] = str(entropy)

                '''hashtags = pp.preprocessing(userTweets['text'])
                countMatrix = vectorizer2.fit_transform(hashtags)
                hashtags = vectorizer.get_feature_names()
                users.at[index, 'hashtags2gram'] = " ".join(hashtags)
                frequency = np.asarray(countMatrix.sum(axis=0))
                users.at[index, 'frequence2'] = " ,".join(map(str, frequency))
                total = frequency.sum()
                proba = frequency / total
                entropy = - np.sum(proba * np.log2(proba))
                users.at[index, 'entropie2hashtags'] = str(entropy)'''

            else:
                indxDelete.append(index)

        except ValueError:
            indxDelete.append(index)
users.drop(users.index[indxDelete])
users =   users[users['nbTweets'] >= 100]


print("Temps pris en secondes : " + str(time.time() - start))
users.to_sql("entropie_clusters", engine, "pldac", if_exists='append')
#mettre append et faire bout par bout
