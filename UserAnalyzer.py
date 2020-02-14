# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:52:42 2020

@author: arnaud
"""

''' 
Class UserAnalyzer pour fournir les indicateurs d'un utilisateur
et définir la classe politique de l'utilisateur

'''

class UserAnalyzer():
    
    def __init__(self):
        self.id = 0
        self.classePolitique = 0
        self.nbFollowers = 0
        self.nbFollow = 0
        self.tweets = 0
        
    def classifyUser(self):
        # analiser ses tweets
        self.classePolitique= 0
        return

