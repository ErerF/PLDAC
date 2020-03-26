#!/usr/bin/env python3
"""
modified from:
    https://github.com/patapizza/pylouvain/blob/master/pylouvain.py
    
@author: Zixuan FENG
         Arnaud DELOL
"""

'''
    Implements the Louvain method.
    Input: a weighted directed graph
    Ouput: a (partition, modularity) pair where modularity is maximum
'''
import json
import mysql.connector
from collections import Counter

# rebuild graph with successive identifiers
def normalize(edges):   
    noeuds=set()
    for e in edges:
        noeuds.add(e[0][0])
        noeuds.add(e[0][1])
    nodes=list(noeuds)
    nodes.sort()
    i = 0
    nodes_ = []
    dict_node_index = {}
    dict_index_node={}
    for n in nodes:
        nodes_.append(i)
        dict_node_index[n] = i
        dict_index_node[i] = n
        i += 1
    edges_ = []
    for e in edges:
        edges_.append(((dict_node_index[e[0][0]], dict_node_index[e[0][1]]), e[1]))
    return (nodes_, edges_, dict_node_index, dict_index_node)

#test normalize()
#print(normalize([(('a','b'),1),(('b','c'),2)]))
#return ([0, 1, 2], [((0, 1), 1), ((1, 2), 2)], {'a': 0, 'b': 1, 'c': 2})


class PyLouvain:
    def __init__(self):
		#fetch data
        f=open("database_info.json")
        db_info=json.load(f)
        u=db_info["user"]
        pwd=db_info["pwd"]
        h=db_info["host"]
        db=db_info["database"]         
        cnx = mysql.connector.connect(user=u, password=pwd, host=h, database=db)
        cursor = cnx.cursor()
        
        edges=[]                
        sql="SELECT source_user_id,target_user_id FROM tweets.user_mentions_0415_0423 LIMIT 1000;"
        try:
            cursor.execute(sql)
            edges = cursor.fetchall()            
        except:
            print("Error: unable to fetch data")
        cursor.close()
        cnx.close()       
        
        #self.nodes, self.edges, self.m
        e_weighted=list(Counter(edges).items())
        self.nodes,self.edges,self.dict_node_index,self.dict_index_node=normalize(e_weighted)
        self.m=len(self.edges)
        
        
        #self.w
        #self.k_i_in, self.k_i_out
        #self.out_edges_of_node, self.in_edges_of_node
        self.w=[0 for n in self.nodes]
        self.k_i_in=[0 for n in self.nodes]    #in_degree
        self.k_i_out=[0 for n in self.nodes]   #out_degree
        self.out_edges_of_node={}
        self.in_edges_of_node={}
        for e in self.edges:
            self.k_i_out[e[0][0]]+=e[1]
            self.k_i_in[e[0][1]]+=e[1]
            if e[0][0]==e[0][1]:
                self.w[e[0][0]]+=e[1]
            
            if e[0][0] not in self.out_edges_of_node:
                self.out_edges_of_node[e[0][0]] = [e]
            else:
                self.out_edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.in_edges_of_node:
                self.in_edges_of_node[e[0][1]] = [e]
            else:
                self.in_edges_of_node[e[0][1]].append(e)    
        
        self.communities = [n for n in self.nodes]
        self.actual_partition = []
        
        
    """
    def apply_method(self):
        network = (self.nodes, self.edges)
        best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        while 1:
            #print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network)
            q = self.compute_modularity(partition)
            partition = [c for c in partition if c]
            #print("%s (%.8f)" % (partition, q))
            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            if q == best_q:
                break
            network = self.second_phase(network, partition)
            best_partition = partition
            best_q = q
        return (self.actual_partition, best_q)  
    """
        
        
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        #self.s_in = [0 for node in network[0]]
        self.s_in = [self.w[node] for node in network[0]]
        self.s_tot_in = [self.k_i_in[node] for node in network[0]]
        self.s_tot_out = [self.k_i_out[node] for node in network[0]]
        """
        for e in network[1]:
            if e[0][0] == e[0][1]: # only self-loops
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1]
        """
        return partition
    
    '''
        Performs the first phase of the method.
        _network: a (nodes, edges) pair
    '''
    def first_phase(self, network):
        # make initial partition
        best_partition = self.make_initial_partition(network)
        while 1:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                # default best community is its own
                best_community = node_community
                best_gain = 0
                # remove _node from its community
                best_partition[node_community].remove(node)
                best_shared_links_in = 0
                if node in self.in_edges_of_node:
                    for e in self.in_edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if self.communities[e[0][0]] == node_community:
                            best_shared_links_in += e[1]
                self.s_in[node_community] -= (best_shared_links_in + self.w[node])
                self.s_tot_in[node_community] -= self.k_i_in[node]
                self.s_tot_out[node_community] -= self.k_i_out[node]
                self.communities[node] = -1
                communities = {} # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    if community in communities:
                        continue
                    communities[community] = 1
                    shared_links = 0
                    if node in self.in_edges_of_node:
                        for e in self.in_edges_of_node[node]:
                            if e[0][0] == e[0][1]:
                                continue
                            if self.communities[e[0][0]] == community:
                                shared_links += e[1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(node, community, shared_links)
                    if gain > best_gain:
                        best_community = community
                        best_gain = gain
                        best_shared_links_in = shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                self.s_in[best_community] += (best_shared_links_in + self.w[node])
                self.s_tot_in[best_community] += self.k_i_in[node]
                self.s_tot_out[best_community] += self.k_i_out[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    
    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''
    #delta Q
    def compute_modularity_gain(self, node, c, k_i_in):
        return k_i_in - (self.s_tot_in[c] * self.k_i_out[node]+self.s_tot_out[c] * self.k_i_in[node]) / self.m
    
    '''
        Yields the nodes adjacent to _node.
        _node: an int
    '''
    def get_neighbors(self, node):
        if node in self.in_edges_of_node:
            for e in self.in_edges_of_node[node]:
                if e[0][0] == e[0][1]: # a node is not neighbor with itself
                    continue
                yield e[0][0]
        if node in self.out_edges_of_node:
            for e in self.out_edges_of_node[node]:
                if e[0][0] == e[0][1]: # a node is not neighbor with itself
                    continue
                yield e[0][1]
        
        
    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''
    """
    #Q
    def compute_modularity(self, partition):
        q = 0
        m2 = self.m * 2
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q
    """






p=PyLouvain()   
network = (p.nodes, p.edges)
#p.make_initial_partition(network)
first=p.first_phase(network)
#print(first)
#print(len(first))
first_not_empty=[f for f in first if f!=[]]
#print(first_not_empty)
#print(len(first_not_empty))

partition={}
for i in range(len(first_not_empty)):
    for n in first_not_empty[i]:
        partition[p.dict_index_node[n]]=i
print(partition)


g=nx.DiGraph(p.edges)  
size = float(len(set(partition.values())))
pos = nx.spring_layout(g)
count = 0.
for com in set(partition.values()) :
    count = count + 1.

"""
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(g, pos, first_not_empty, node_size = 20,
                                node_color = str(count / size))
"""

    nx.draw_networkx_nodes(g, pos, first_not_empty[int(count-1)], node_size = 20, node_color = str(count / size))

nx.draw_networkx_edges(g, pos, alpha=0.5)
#plt.savefig("Images\louvain_first1000_directed.png")
plt.show()  


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''
    @classmethod
    def from_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes[n[0]] = 1
            nodes[n[1]] = 1
            w = 1
            if len(n) == 3:
                w = int(n[2])
            edges.append(((n[0], n[1]), w))
        # rebuild graph with successive identifiers
        nodes_, edges_ = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
        return cls(nodes_, edges_)

    '''
        Initializes the method.
        _nodes: a list of ints
        _edges: a list of ((int, int), weight) pairs
    '''
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        # precompute m (sum of the weights of all links in network)
        #            k_i (sum of the weights of the links incident to node i)
        self.m = 0
        self.k_i = [0 for n in nodes]
        self.edges_of_node = {}
        self.w = [0 for n in nodes]
        for e in edges:
            self.m += e[1]
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1] # there's no self-loop initially
            # save edges by node
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # access community of a node in O(1) time
        self.communities = [n for n in nodes]
        self.actual_partition = []


    '''
        Applies the Louvain method.
    '''
    def apply_method(self):
        network = (self.nodes, self.edges)
        best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        while 1:
            #print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network)
            q = self.compute_modularity(partition)
            partition = [c for c in partition if c]
            #print("%s (%.8f)" % (partition, q))
            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            if q == best_q:
                break
            network = self.second_phase(network, partition)
            best_partition = partition
            best_q = q
        return (self.actual_partition, best_q)

    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''
    def compute_modularity(self, partition):
        q = 0
        m2 = self.m * 2
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''
    def compute_modularity_gain(self, node, c, k_i_in):
        return 2 * k_i_in - self.s_tot[c] * self.k_i[node] / self.m

    '''
        Performs the first phase of the method.
        _network: a (nodes, edges) pair
    '''
    def first_phase(self, network):
        # make initial partition
        best_partition = self.make_initial_partition(network)
        while 1:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                # default best community is its own
                best_community = node_community
                best_gain = 0
                # remove _node from its community
                best_partition[node_community].remove(node)
                best_shared_links = 0
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])
                self.s_tot[node_community] -= self.k_i[node]
                self.communities[node] = -1
                communities = {} # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    if community in communities:
                        continue
                    communities[community] = 1
                    shared_links = 0
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if e[0][0] == node and self.communities[e[0][1]] == community or e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_links += e[1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(node, community, shared_links)
                    if gain > best_gain:
                        best_community = community
                        best_gain = gain
                        best_shared_links = shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    

    '''
        Builds the initial partition from _network.
        _network: a (nodes, edges) pair
    '''
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_in = [0 for node in network[0]]
        self.s_tot = [self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]: # only self-loops
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1]
        return partition

    '''
        Performs the second phase of the method.
        _network: a (nodes, edges) pair
        _partition: a list of lists of nodes
    '''
    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]
        # relabelling communities
        communities_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        self.communities = communities_
        # building relabelled edges
        edges_ = {}
        for e in network[1]:
            ci = self.communities[e[0][0]]
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]
        edges_ = [(k, v) for k, v in edges_.items()]
        # recomputing k_i vector and storing edges by node
        self.k_i = [0 for n in nodes_]
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)

'''
    Rebuilds a graph with successive nodes' ids.
    _nodes: a dict of int
    _edges: a list of ((int, int), weight) pairs
'''
def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes = list(nodes.keys())
        nodes.sort()
        i = 0
        nodes_ = []
        d = {}
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1
        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))
        return (nodes_, edges_)
"""