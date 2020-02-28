# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:14:59 2020

@author: Zixuan FENG
         Arnaud DELOL
"""

"""
Implementer l'algo de Louvain
Entree: un graphe pondere non-oriente
Sortie: (partition, modularite) ou la modularite est maximal
"""
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import json
import mysql.connector

#recuperer les noeuds et les arcs
def nodes_edges():
    f=open("database_info.json")
    db_info=json.load(f)
    u=db_info["user"]
    pwd=db_info["pwd"]
    h=db_info["host"]
    db=db_info["database"] 
        
    cnx = mysql.connector.connect(user=u, password=pwd, host=h, database=db)
    cursor = cnx.cursor()
    #nodes=set()
    edges=[]
    
    sql="SELECT source_user_id,target_user_id FROM tweets.user_mentions_0415_0423 LIMIT 1000;"
    try:
        cursor.execute(sql)
        edges = cursor.fetchall()
    except:
        print("Error: unable to fetch data")
    cursor.close()
    cnx.close()
    """
    for e in edges:
        nodes.add(e[0])
        nodes.add(e[1])
    """
    return edges
      
#construiure le graphe
def create_graph(edges):
    ct=Counter(edges)
    l=[]
    g=nx.DiGraph()
    for k in ct.keys():
        l.append((k[0],k[1],ct[k]))    
    g.add_weighted_edges_from(l)
    
    return g
    
def draw_graph(g):
    #nx.draw_networkx_edge_labels(g,nx.spring_layout(g))
    #nx.draw_networkx(g)
    pos=nx.spring_layout(g)
    nx.draw(g, pos, alpha=0.5)
    plt.show()


edges=nodes_edges()
#g=create_graph(edges)
#draw_graph(g)


ct=Counter(edges)
l=[]
g=nx.Graph()
for k in ct.keys():
    l.append((k[0],k[1],ct[k]))
#g.add_nodes_from(nodes)
g.add_weighted_edges_from(l)

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
plt.savefig("Images\louvain_first1000.png")
plt.show()

"""
#2 facons a dessiner:
#1.
#nx.draw_networkx(g)
#2.
nx.draw(g)
#plt.savefig("louvain.png")  #enregistrer dans un fichier
plt.show()
"""
