# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:20:38 2020

@author: Zixuan FENG
"""
import json
import mysql.connector
import csv
from collections import Counter


def get_reactions():
    f=open("database_info.json")
    db_info=json.load(f)
    u=db_info["user"]
    pwd=db_info["pwd"]
    h=db_info["host"]
    db=db_info["database"]         
    cnx = mysql.connector.connect(user=u, password=pwd, host=h, database=db)
    cursor = cnx.cursor()
    #sql="SELECT * FROM tweets.tweets_0415_0423 LIMIT 1000"          
    sql="SELECT user_id,in_reply_to_user_id,quoted_user_id,retweeted_user_id FROM tweets.tweets_0415_0423 WHERE (in_reply_to_user_id IS NOT NULL OR quoted_user_id IS NOT NULL OR retweeted_user_id IS NOT NULL);"
    res=[]
    try:
        cursor.execute(sql)
        res = cursor.fetchall()               
    except:
        print("Error: unable to fetch data")
    cursor.close()
    cnx.close()  
    
    reactions=[]
    #f=open("./Louvain/interaction_1000.csv",'w',encoding="utf-8")
    #f.write("user_id in_reply_to_user_id quoted_user_id retweeted_user_id\r\n")
    for r in res:
        reactions.append((r[0],r[1],r[2],r[3]))
        #f.write(str(r[0])+" "+str(r[1])+" "+str(r[2])+" "+str(r[3])+"\r\n")
    #f.close() 
    return reactions
    
'''
    return [(1029413485, 983300684),(.,.),...]
'''
def merge_reactions(reactions):
    edges=[]
    for r in reactions:
        if r[1]!=None:
            edges.append((r[0],r[1]))
        if r[2]!=None:
            edges.append((r[0],r[2]))
        if r[3]!=None:
            edges.append((r[0],r[3]))
    return edges
    
def save_undirected_edges(edges):
    undirect=dict()
    for e in edges:
        if e in undirect:
            undirect[e]+=1
        else:
            if (e[1],e[0]) in undirect:
                undirect[(e[1],e[0])]+=1
            else:
                undirect[e]=1
                
    f=open("./Louvain/undirected_edges_1000.csv",'w',encoding="utf-8")
    f.write("source target weight\r\n")
    for u in undirect.keys():
        f.write(str(u[0])+" "+str(u[1])+" "+str(undirect[u])+"\r\n")
    f.close() 
    
    return undirect
    
def save_directed_edges(edges):
    
    cpt=Counter(edges)   
    direct=dict(cpt)
    f=open("./Louvain/directed_edges_1000.csv",'w',encoding="utf-8")
    f.write("source target weight\r\n")
    for c in cpt.keys():
        f.write(str(c[0])+" "+str(c[1])+" "+str(cpt[c])+"\r\n")
    f.close() 
    
    return direct
    
# rebuild graph with successive identifiers
def normalize(edges): 
    i=0
    edges_=[]
    dict_node_index = {}
    dict_index_node={}
    for e in edges:
        if e[0] not in dict_node_index:
            dict_node_index[e[0]]=i
            dict_index_node[i]=e[0]
            i+=1
        if e[1] not in dict_node_index:
            dict_node_index[e[1]]=i
            dict_index_node[i]=e[1]
            i+=1
        edges_.append((dict_node_index[e[0]],dict_node_index[e[1]]))
    
    return (i, edges_, dict_node_index, dict_index_node)

#test normalize()
#print(normalize([(('a','b'),1),(('b','c'),2)]))
#return ([0, 1, 2], [((0, 1), 1), ((1, 2), 2)], {'a': 0, 'b': 1, 'c': 2})



    
  
reactions=get_reactions()
edges=merge_reactions(reactions) #edges=[('b','c'),('a','b')]
#nbNodes=3, normEdges=[(0,1),(2,0)]
#dni={'b':0,'c':1,'a':2}
#din={0:'b',1:'c',2:'a'}
nbNodes,normEdges,dni,din=normalize(edges) 

#undirect=save_undirected_edges(edges)
#direct=save_directed_edges(edges)

import Louvain_directed
import Louvain_undirected
import time

start=time.time()
print("in Louvain_Di()")
pDi=Louvain_directed.Louvain_directed(nbNodes,normEdges)  
print("out PyLouvain()",time.time()-start)
#networkDi = (pDi.nodes, pDi.edges)
#print("network initialized")
pDi.apply_method()

print("============================================")

start=time.time()
print("in Louvain_Undi()")
pUndi=Louvain_undirected.Louvain_undirected(nbNodes,normEdges)   
print("out PyLouvain()",time.time()-start)
#networkUndi = (pUndi.nodes, pUndi.edges)
#print("network initialized")
pUndi.apply_method()
















