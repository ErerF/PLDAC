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

#recuperer les listes de noeuds et d'arcs
nodes=[1,2,3,4]  #nodes=load()
#edges=[(1,2,2),(1,3,1),(2,3,3)]  #edges=load()
edges=[(1,2),(1,3),(2,3)]

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
"""
nx.draw(g)
#plt.savefig("louvain.png")  #enregistrer dans un fichier
plt.show()
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