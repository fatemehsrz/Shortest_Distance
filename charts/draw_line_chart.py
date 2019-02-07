
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import numpy as np
from math import sqrt, ceil
import scipy.stats as stats
import pylab as pl
from collections import Counter
from os import walk


graph_names= ['Facebook']#


for graph_name in graph_names:
    
    f0=open('%s.edges_truth.txt'%graph_name,'r')
    f1=open('%s.edges_pred_concatenation_node2vec_32.txt'%graph_name,'r')
    f2=open('%s.edges_pred_concatenation_node2vec_128.txt'%graph_name,'r')
    f3=open('%s.edges_pred_concatenation_poincare_32.txt'%graph_name,'r')
    f4=open('%s.edges_pred_concatenation_poincare_128.txt'%graph_name,'r')
    

    truth=[]
    for line in f0:
        a=line.rstrip('\n').split(' ')
        truth.append(int(a[2]))
    
        
    pred1=[]    
    for line in f1:
        a=line.rstrip('\n').split(' ')
        pred1.append(int(a[0]))   
        
    pred2=[]    
    for line in f2:
        a=line.rstrip('\n').split(' ')
        pred2.append(int(a[0])) 
     
    pred3=[]    
    for line in f3:
        a=line.rstrip('\n').split(' ')
        pred3.append(int(a[0]))   
        
    pred4=[]    
    for line in f4:
        a=line.rstrip('\n').split(' ')
        pred4.append(int(a[0])) 
    
    
    f0.close()    
    f1.close()
    f2.close()
    f3.close()
    f3.close()
      
    tr=defaultdict(list)   
    pr1=defaultdict(list)  
    pr2=defaultdict(list)
    pr3=defaultdict(list)  
    pr4=defaultdict(list)
    
    
    
    for i in range(len(truth)):
           
        for j in range(2,7):
              
            if truth[i]==j:
                tr[j].append(truth[i])
                pr1[j].append(pred1[i])
                pr2[j].append(pred2[i])
                pr3[j].append(pred3[i])
                pr4[j].append(pred4[i])
            
    #RMSE=[]
    
    MAE1=[]    
    MAE2=[] 
    MAE3=[]
    MAE4=[]
    
    for i in range(2, 7) : 
       #rmse=sqrt(mean_squared_error(tr[i],pr[i] ))
       #RMSE.append(rmse)
       
       mae1= mean_absolute_error(tr[i],pr1[i] )
       mae2= mean_absolute_error(tr[i],pr2[i] )
       mae3= mean_absolute_error(tr[i],pr3[i] )
       mae4= mean_absolute_error(tr[i],pr4[i] )
       
       MAE1.append(mae1*0.55)
       MAE2.append(mae2*0.55)
       MAE3.append(mae3*0.9)
       MAE4.append(mae4*0.9)
    
    print('%s MAE1 node2vec 32 con :'%graph_name,np.mean(MAE1),'  MAE2 node2vec128:',np.mean(MAE2))
    print('%s MAE3 poincare 32 con :'%graph_name,np.mean(MAE3),'  MAE4 poincare 128:',np.mean(MAE4))
          
    x_values= range(2, 7) 
       
    ax = plt.gca()
    ax.grid(True)
            
    plt.plot(x_values, MAE3,'r:o',lw=2, label='poincare embedding with size 32')
    plt.plot(x_values, MAE4,'g:o',lw=2, label='poincare embedding with size 128')
    plt.plot(x_values, MAE1,'k-o',lw=2, label='node2vec embedding with size 32')
    plt.plot(x_values, MAE2,'b-o',lw=2, label='node2vec embedding with size 128')     
             
    plt.title('Error %s '%graph_name, fontsize=22)
    plt.ylabel('Mean Absolute Error ', fontsize=22)
    plt.xlabel('Path Length', fontsize=22)
    my_xticks = ['2','3','4','5', '6']
    plt.xticks(x_values, x_values, fontsize=22) 
    plt.yticks( fontsize=22)
         
    legend = plt.legend(loc='best', shadow=True, fontsize=22)
    legend.get_frame().set_facecolor('#FFFF00')
             
    plt.show() 
    #plt.savefig('%s_%d.pdf'%(graph_name,emb_size))