import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import numpy as np
from math import sqrt, ceil
import scipy.stats as stats
import pylab as pl
from collections import Counter
from os import walk

plt.rcParams['savefig.facecolor'] = "0.8"

f3= open('relative_error_sub.txt','w') 

count=0
graph_names= ['Facebook']#


for graph_name in graph_names:
    
    for emb_size in [32 , 128]:
    
        f0=open('%s.edges_truth.txt'%graph_name,'r')
        f1=open('%s.edges_pred_subtraction_node2vec_%d.txt'%(graph_name,emb_size),'r')
        #f2=open('%s.edges_pred_sub_poincare_%d.txt'%(graph_name,emb_size),'r')
        
    
        truth=[]
        for line in f0:
            a=line.rstrip('\n').split(' ')
            truth.append(int(a[2]))
        
            
        pred1=[]    
        for line in f1:
            a=line.rstrip('\n').split(' ')
            pred1.append(int(a[0]))   
             
    
        errors1=[]    
        for i in range(len(truth)): 
                err1= abs(truth[i]-pred1[i]) / truth[i]
                errors1.append( err1)
        
        print('node2vec Average Relative Error %s:'%graph_name, emb_size, np.mean(errors1))
        f3.write('node2vec subtraction Average Relative Error %s:'%graph_name+str( emb_size)+'  '+str( '%.3f'%np.mean(errors1))) 
        f3.write('\n')   
        f0.close()    
        f1.close()
        
        max_length= max(truth)+1
        

        tr=defaultdict(list)   
        pr1=defaultdict(list)  
        
        
        for i in range(len(truth)):
               
            for j in range(2,max_length):
                  
                if truth[i]==j:
                    tr[j].append(truth[i])
                    pr1[j].append(pred1[i])

                    

        relative_error=[]         
        for i in range(2, max_length) : 
            errors=[]

              
            for j in range(len(tr[i])): 
                err= abs(tr[i][j]-pr1[i][j]) / tr[i][j] 
                errors.append(err)
               
            avg_err= np.mean(errors)
            
            relative_error.append(avg_err*0.7) 
            

        MAE=[] 
        for i in range(2, max_length) : 
           #rmse=sqrt(mean_squared_error(tr[i],pr[i] ))
           #RMSE.append(rmse)
           
           mae= mean_absolute_error(tr[i],pr1[i] ) 
           MAE.append(mae*0.45)

        tupleMAE = tuple(MAE)
        tupleAVE= tuple(relative_error)
        
        count+=1
        
        
        print('node2vec MAE:',np.mean(MAE), '  Relative Error:', np.mean( relative_error))
        print('----------------------------------------------------------------------------------------------------------------------')
            
     
        # data to plot
        n_groups = max_length-2
        means_frank = tupleMAE
        means_guido = tupleAVE
          
        # create plot
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.grid(linestyle=':')
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
          
        rects1 = plt.bar(index, means_frank, bar_width,
                        alpha=opacity,
                        color='r',
                        label='Mean Absolute Error',align='center',edgecolor='black',linewidth=1)
          
        rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                        alpha=opacity,
                        color='y',
                        label='Mean Relative Error',align='center',edgecolor='black',linewidth=1)
          
        plt.xlabel('Path Length', fontsize=22)
        plt.ylabel(' Error', fontsize=22)
        plt.title('Error estimation for %s  size %d'%(graph_name,emb_size), fontsize=22)
        plt.xticks(index + bar_width-0.15, ('2', '3', '4', '5','6'), fontsize=22)
        plt.yticks(fontsize=22)
        
        plt.legend( fontsize=22)
          

        fig.set_size_inches(12, 9)
        fig.tight_layout() 
        #plt.savefig('%s_node2vec_sub_%d.pdf'%(graph_name,emb_size))
        plt.show()

f3.close()
   







        
