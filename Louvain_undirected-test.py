# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:57:45 2020

@author: DELL
"""


import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import json
import mysql.connector
import copy

#recuperer les noeuds et les arcs
f=open("database_info.json")
db_info=json.load(f)
u=db_info["user"]
pwd=db_info["pwd"]
h=db_info["host"]
db=db_info["database"] 
    
cnx = mysql.connector.connect(user=u, password=pwd,
                              host=h,
                              database=db)
cursor = cnx.cursor()
nodes=set()
edges=[]
rows=[]

sql="SELECT source_user_id,target_user_id FROM tweets.user_mentions_0415_0423 LIMIT 1000;"
try:
    cursor.execute(sql)
    rows = cursor.fetchall()
except:
    print("Error: unable to fetch data")
cursor.close()
cnx.close()
  
edges=copy.deepcopy(rows)
for r in rows:
    nodes.add(str(r[0]))
    nodes.add(str(r[1]))   



#g=nx.DiGraph(nx.to_networkx_graph(edges))  #wrong!!
g=nx.DiGraph(edges)  
"""
print(g.nodes())
print(g.edges())
"""
print(len(g.nodes()))
print(len(nodes))

print(len(g.edges()))
print(len(edges))
"""
#nx.draw_networkx(g)

c=Counter(edges)
#print(sum(c.values()))
print(len(c))
print(len(set(edges)))

print(g.edges()-c.keys())

"""



"""

#"SELECT DISTINCT source_user_id FROM tweets.user_mentions_0415_0423;"
sql="SELECT DISTINCT source_user_id FROM tweets.user_mentions_0415_0423 LIMIT 1000;"
try:
    cursor.execute(sql)
    rows = cursor.fetchall()
    
except:
    print("Error: unable to fetch data")

for r in rows:
    nodes.add(str(r[0]))

#SELECT DISTINCT target_user_id FROM tweets.user_mentions_0415_0423;
sql="SELECT DISTINCT target_user_id FROM tweets.user_mentions_0415_0423 LIMIT 1000;"
try:
    cursor.execute(sql)
    rows = cursor.fetchall()
    for r in rows:
        nodes.add(str(r[0]))  
except:
    print("Error: unable to fetch data")

nodes=list(nodes)
print(len(nodes))

"""



"""

#construiure le graphe
#g=nx.DiGraph()
g=nx.Graph()
g.add_nodes_from(nodes)
#g.add_weighted_edges_from(edges)
g.add_edges_from(edges)
#2 facons a dessiner:
#1.
#nx.draw_networkx(g)
#2.
nx.draw(g)
#plt.savefig("louvain.png")  #enregistrer dans un fichier
plt.show()
"""





#Louvain--undirected graph

"""
#Louvain: https://pypi.org/project/python-louvain/
import community
partition=community.best_partition(g)
size = float(len(set(partition.values())))
pos = nx.spring_layout(g)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(g, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(g, pos, alpha=0.5)
plt.show()
"""


