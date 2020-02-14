# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:02:27 2020

@author: arnaud
"""


'''
Objet qui cree la connection a la base de donnees
Qui effectue les requetes et renvoie le resultat
'''

import mysql.connector


class CNXManager():
    
    def __init__(self,user,password,database):
        self.cnx = mysql.connector.connect(user=user, password=password,
                              host='127.0.0.1',
                              database=database)
        self.cursor = self.cnx.cursor()
        
    # execute la requete et renvoie le cursor et les rows
    def executeSelectQuery(self,query):
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows
    
    def closeCNX(self):
        self.cursor.close()
        self.cnx.close() 
        