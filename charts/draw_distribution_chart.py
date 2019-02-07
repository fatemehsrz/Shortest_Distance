import networkx as nx
from scipy.stats import entropy
#from scipy.stats import  histogram
import numpy as np
import scipy.stats as stats
import pylab as plt
from collections import Counter

graph_names= ['Facebook']  #'facebook','youtube','BlogCatalog''facebook(NIPS)'
for graph_name in graph_names:
    
    f2= open('%s.edges_truth.txt'%graph_name,'r')
    
    truth=[]
    for line in f2:
        a=line.rstrip('\n').split(' ')
        truth.append(int(a[2]))


    counter_dis= dict(Counter(truth))  
    
    print(max(truth))
    prob=[]
    total=0
    for i in range(2,max(truth)+1):
        
        total+=counter_dis[i]
        print('path', i, 'count', counter_dis[i])
        print('total:',total)
        
    for i in range(2,max(truth)+1):
        pr= counter_dis[i]/total
        print(pr) 
        prob.append(pr)
        
    prob_distribution= tuple(prob) 
        
   
    

    print(prob)
    print(list(range(2, max(truth)+1)))
    

    # data to plot
    n_groups = max(truth)-1
  
    # create plot
    fig, ax = plt.subplots()
    #ax.grid(True)
    #ax.grid(linestyle=':')
    index = np.arange(n_groups)
    bar_width = 0.7
    opacity = 0.9
       
    rects1 = plt.bar(index, prob_distribution , bar_width,
                    alpha=opacity,
                    color='b', align='center',edgecolor='black',linewidth=1.1) #label='Probablity Distribtion'
       
       
    plt.xlabel('Path Length', fontsize=22)
    plt.ylabel('Probability ', fontsize=22)
    plt.title('%s distances distribution'%graph_name, fontsize=22)
    plt.xticks(index+(bar_width)-0.7, ('2', '3', '4', '5','6'), fontsize=22)
    plt.yticks( fontsize=22)
    plt.legend(fontsize=22)
      
 
    #fig.set_size_inches(12, 9)Path Length Distribution in the Training Set for
    fig.tight_layout() 
    plt.savefig('%s_test_distribution.pdf'%graph_name)
    plt.show()
   
    
    
    