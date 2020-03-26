
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
import re
from sqlalchemy import create_engine
from pathlib import Path

def longueur(tweet):
    return len(tweet)

def inreply(tweet):
    if str(tweet) != 'nan':
        return 1
    else:
        return 0

def presenceLien(tweet):
    regexp = re.compile(r'http')
    if regexp.search(tweet):
        return 1
    else:
        return 0




pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',-1)


def analyseSelonEntropie(entropie):

    nomDossier = 'Dossier' + str(entropie)
    try:
        Path(nomDossier).mkdir()
    except:
        print('Dossier deja cree')
    window = 200

    engine = create_engine("mysql+mysqlconnector://root:root@127.0.0.1/pldac")
    users = pd.read_sql_query(
        "select user_id,followers_count,friends_count,nbTweets,hashtags1gram,frequence1,entropie1hashtags from entropie_tweets2 where entropie1hashtags != '' and entropie1hashtags  <= " + str(entropie) +"limit 10000000", engine)
    entropie = users['entropie1hashtags'].values
    temp = []
    '''
    plt.figure(window)
    window += 100
    plt.scatter(users['entropie1hashtags'],users['nbTweets'])
    plt.draw()
    plt.show()
    '''
    users['longueurMoyenneTweet'] = 0
    users['nbRetweetsMoyen'] = 0


    for index,user in users.iterrows():
        userTweets = pd.read_sql_query(
            "select text,retweet_count ,in_reply_to_status_id from tweets_0415_0423 where user_id =  " + str(user['user_id']), engine)
        userTweets['longueur'] = 0
        userTweets['inreply'] = 0
        userTweets['presenceLien'] = 0
        userTweets['inreply'] = userTweets['in_reply_to_status_id'].apply(inreply)
        userTweets['longueur'] = userTweets['text'].apply(longueur)
        userTweets['presenceLien'] = userTweets['text'].apply(presenceLien)
        users.at[index, 'longueurMoyenneTweet'] = userTweets['longueur'].mean()
        users.at[index, 'nbRetweetsMoyen'] = userTweets['retweet_count'].mean()
        nbTweet = users.at[index, 'nbTweets']
        users.at[index, 'nbinReplyMoyen'] = userTweets['inreply'].sum()/nbTweet
        users.at[index, 'ratioLienParTweet'] = userTweets['presenceLien'].sum() / nbTweet

    print(users)
    plt.figure(window)
    window += 100
    plt.scatter(users['nbTweets'],users['longueurMoyenneTweet'])
    plt.title("Longueur moyenne tweet selon le nb de tweets")
    plt.xlabel("Nb de tweet du user")
    plt.ylabel("Longueur moyenne des tweets")
    plt.draw()
    chemin = nomDossier+"/longueur selon nbTweets.png"
    plt.savefig(chemin)
    plt.show()

    plt.figure(window)
    window += 100
    plt.scatter(users['nbTweets'],users['nbRetweetsMoyen'])
    plt.title("retweet moyen des tweets d'un user selon le nb de tweets")
    plt.xlabel("Nb de tweets du user")
    plt.ylabel("Nb de retweets sur les tweet du user")
    plt.draw()
    chemin = nomDossier+"/NbRetweetsMoyen selon nbTweets.png"
    plt.savefig(chemin)
    plt.show()

    plt.figure(window)
    window += 100
    entro = users['entropie1hashtags'].values
    entro= np.array([float(i) for i in entro])
    plt.scatter(entro,users['longueurMoyenneTweet'])
    plt.title("longueur moyenne des tweets selon l'entropie")
    plt.xlabel("Entropie")
    plt.ylabel("Longueur moyenne d'un tweet")
    plt.draw()
    chemin = nomDossier+"/longueur_selon_entropie.png"
    plt.savefig(chemin)
    plt.show()
    print("done")

    print(users.shape)

analyseSelonEntropie(2.5)