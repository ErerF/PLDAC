

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
import pandas as pd
from sqlalchemy import create_engine

class CNXManager():
    
    def __init__(self,user,password,database):
        self.engine = create_engine("mysql+mysqldb://root:root@localhost/pldac")
        self.cursor = self.cnx.cursor()
        
    # execute la requete et renvoie le cursor et les rows
    def executeSelectQuery(self,query):
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows
    
    def closeCNX(self):
        self.cursor.close()
        self.cnx.close() 
        