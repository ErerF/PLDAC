#!/usr/bin/env python3
"""
modified from:
    https://github.com/patapizza/pylouvain/blob/master/pylouvain.py
    
@author: Zixuan FENG
"""


import time


'''
    Implements the Louvain method.
    Input:  number of nodes
            a list of undirected edges, weight=1 for each
    Ouput: a (partition, modularity) pair where modularity is maximum
'''

class Louvain_undirected:
    def __init__(self,nbNodes,edges):    
        '''
        define:
            self.m, self.nodes, self.edges
        '''
        self.m=len(edges)
        self.nodes=[i for i in range(nbNodes)]
        self.edges={}
        for e in edges:
            if e[0]>e[1]:
                i=e[1]
                j=e[0]
            else:
                i=e[0]
                j=e[1]
                
            if (i,j) in self.edges:
                self.edges[(i,j)]+=1
            else:
                self.edges[(i,j)]=1
        self.edges=list(self.edges.items())
        
        '''
        define:
            self.w, self.k_i, self.edges_of_node
        '''
        self.w=[0 for n in self.nodes]
        self.k_i=[0 for n in self.nodes]
        self.edges_of_node={}
        for e in self.edges:
            self.k_i[e[0][0]]+=e[1]
            self.k_i[e[0][1]]+=e[1]
            if e[0][0]==e[0][1]:
                self.w[e[0][0]]+=e[1]
            
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)    
        
        self.communities = [n for n in self.nodes]
        self.actual_partition = []
        
    def apply_method(self):
        f=open("./Louvain/undirected_applyMethod.txt",'w',encoding="utf-8")
        
        network = (self.nodes, self.edges)
        #best_partition = [[node] for node in network[0]]
        best_q = -1
        
        i = 0
        while 1:
            i += 1
            
            print("cycle ",str(i-1))
            print("  in first_phase()")            
            start=time.time()
            partition = self.first_phase(network)    
            print("  out first_phase()",time.time()-start)
            
            f.write("cycle "+str(i-1)+" phase 1:\n    "+str(partition)+"\n")  
            q = self.compute_modularity(partition)
            f.write("    q="+str(q)+"\n\n") 
            
            partition = [c for c in partition if c] #remove empty communities
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
            
            print("  in second_phase()")            
            start=time.time()
            network = self.second_phase(network, partition)
            print("  out second_phase()",time.time()-start)
            f.write("cycle "+str(i-1)+" phase 2:\n    "+str(network[0])+"\n\n    "+str(network[1])+"\n\n\n")  
            
            #best_partition = partition
            best_q = q
            
        f.write("\n\n\n\n\n\n\n"+str(self.actual_partition))
        f.close()
        return (self.actual_partition, best_q)  
    
        
        
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_in = [2*self.w[node] for node in network[0]]
        self.s_tot = [self.k_i[node] for node in network[0]]
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
                best_shared_links = 0
                if node in self.edges_of_node:
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if (e[0][0]==node and self.communities[e[0][1]] == node_community) or (e[0][1]==node and self.communities[e[0][0]] == node_community):
                            best_shared_links += e[1]
                self.s_in[node_community] -= 2*(best_shared_links + self.w[node])
                self.s_tot[node_community] -= self.k_i[node]
                
                self.communities[node] = -1
                communities = {} # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    
                    if community in communities:
                        continue
                    
                    communities[community] = 1
                    shared_links = 0
                    if node in self.edges_of_node:
                        for e in self.edges_of_node[node]:
                            if e[0][0] == e[0][1]:
                                continue
                            if (e[0][0]==node and self.communities[e[0][1]] == community) or (e[0][1]==node and self.communities[e[0][0]] == community):
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
                self.s_in[best_community] += 2*(best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1
                    
            if not improvement:
                break
        
        return best_partition

    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''    
    #Q
    def compute_modularity(self, partition):
        q=0
        m2=self.m*2 
        
        #for each community
        for i in range(len(partition)):
            q+=self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q
    
    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''
    #delta Q
    def compute_modularity_gain(self, node, c, k_i_in):
        return 2*k_i_in - self.s_tot[c] * self.k_i[node] / self.m
    
    '''
        Yields the nodes adjacent to _node.
        _node: an int
    '''
    def get_neighbors(self, node):
        if node in self.edges_of_node:
            for e in self.edges_of_node[node]:
                if e[0][0] == e[0][1]: # a node is not neighbor with itself
                    continue
                if e[0][0] == node:
                    yield e[0][1]
                if e[0][1] == node:
                    yield e[0][0]
        
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