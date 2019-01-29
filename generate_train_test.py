import networkx as nx
from math import sqrt, log, log2, ceil
import random
import time
import numpy as np

def generate_pairs(graph_name='facebook.edges' ): 
     
    
    G=nx.read_edgelist("./data/%s.edges"%graph_name) 
    nodes= list(G.nodes())
    
    
    edges= list(G.edges())
         
    num_nodes= len(nodes)
    num_edges= len(edges)
    
    digit= int(ceil(log(num_nodes+1, 10)))
    
    print(digit)
            
    if digit<5:
            
        k1=100
        k2=11
    else:
        k1= 3
        k2= 1
           
    print('k1:',k1,'k2:',k2 )
    
    k_node1=[]
            
    ################ selecting k nodes ###############
            
    for i in range(k1):
        
        n1=random.choice(nodes)
        
        k_node1.append(n1)
     
    ############### remain nodes ##########################
          
    remain_nodes= list(set(nodes)-set(k_node1))
       
    print('number of all nodes:',num_nodes)
    print('number of remaining  nodes:',len(remain_nodes))
 
    ########################### train pairs ################
    
    train_set=[]
    y_train=[]
    
    count=0

    for a in k_node1:
        for b in remain_nodes:
       
            if nx.has_path(G, a, b):
                sPath=nx.shortest_path(G, a, b)
            
                length=1
                for i in range(len(sPath)-1):
                    train_set.append((sPath[0],sPath[i+1], length) )
    
                    length+=1
                    
            
        count+=1
        
        print('train:',count)

    f4= open('./data/%s_train.txt'%(graph_name), 'w')
    
    for i in range(len(train_set)): 
        
        if (6< train_set[i][2] <15):
            f4.write(str(train_set[i][0])+' '+str(train_set[i][1])+' '+str(train_set[i][2]) )
            f4.write('\n')
            
            y_train.append(train_set[i][2])  
            
    f4.close()
           
    print('training pairs:',len(y_train)) 
    print('training max length:',max(y_train)) 
    print('training min length:',min(y_train))
    print('==========================00================================')
    
    
    f1=open('./data/%s_statistics.txt'%(graph_name),'w')    
    
    f1.write('training pairs:'+str(len(y_train)))
    f1.write('\n')
    f1.write('training max length:'+str(max(y_train)))
    f1.write('\n')
    f1.write('training min length:'+str(min(y_train)))
    f1.write('\n')
    
    
    ############################ test pairs #####################
    
    test_set=[]
    k_node2=[]
    
    for i in range(k2):
        
        n2=random.choice(remain_nodes)
        k_node2.append(n2)
    
    remain_nodes2= list(set(remain_nodes)-set(k_node2))
    
    
    count=0
   
    for s in k_node2:
        for d in remain_nodes2:
            
            if nx.has_path(G, s, d):
                sPath=nx.shortest_path(G, s, d) 
                length=1
                for i in range(len(sPath)-1):
                    
                    #print(sPath[0],sPath[i+1], length)
                    test_set.append((sPath[0],sPath[i+1], length) )
    
                    length+=1

        count+=1
        
        print('test:',count)  
    
    
    X_test=[]
    y_test=[]
    
    f2= open('./data/%s_test.txt'%(graph_name), 'w')         
    for i in range(len(test_set)): 
        
        if (6< test_set[i][2] <15):
             
            f2.write(str(test_set[i][0])+' '+ str(test_set[i][1]) +' '+ str(test_set[i][2]) )
            f2.write('\n')
            
            y_test.append(test_set[i][2])
          
    f2.close()
                
    print('number of test pairs:',len(y_test))
    print('test max length:',max(y_test))
    print('test min length:',min(y_test))
    print('============================================================')
    
    f1.write('number of test pairs:'+str(len(y_test)))
    f1.write('\n')
    f1.write('test max length:'+str(max(y_test)))
    f1.write('\n')
    f1.write( 'test min length:'+str(min(y_test))) 
    f1.write('\n') 
    f1.close()  
    

graph_list=[ 'Facebook.edges', 'Youtube.edges']  # 'Facebook.edges' ,

for g in graph_list:

     generate_pairs(g)
    
    