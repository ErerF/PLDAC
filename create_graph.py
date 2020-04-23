# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:52:59 2020

@author: Zixuan FENG
"""

#extract result
f=open("./Louvain/3-din.txt",'r',encoding="utf-8")
lines=f.read()
f.close()
dict_index_node=eval(lines) #string->dict

f=open("./Louvain/3-directed_applyMethod.txt",'r',encoding="utf-8")
res_di=f.readlines()[-1]
f.close()
res_di=eval(res_di) #string->list

#create .csv file
f=open("./Louvain/4-directed_node.csv",'w',encoding="utf-8")
f.write("Id Label Modularity_Class\r\n")
for i in range(len(res_di)):
    for j in range(len(res_di[i])):
        alias=res_di[i][j]
        f.write(str(alias)+" "+str(dict_index_node[alias])+" "+str(i)+"\r\n")
f.close()
print("fini")
        
        
        
        