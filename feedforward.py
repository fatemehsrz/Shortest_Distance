import networkx as nx
import csv
from gensim.models import Word2Vec
import os
import numpy as np
import random
from math import sqrt, log, log2, ceil
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import EarlyStopping, CSVLogger

from sklearn.metrics import mean_squared_error, mean_absolute_error
import gensim.models.keyedvectors as word2vec
import matplotlib.pyplot as plt
from os import walk
import time


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

############################################# classification network ################################
 
class concatenation_node2vec(): 
    
         
    def init_model(self,emb_size):
        
        print ('Compiling Model ... ')
        model = Sequential()
        model.add(Dense(emb_size, input_dim=2*emb_size)) # 50
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        dense_size= int(0.2*emb_size)
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        
        model.add(Dense(1))         # longest path
        model.add(Activation('softplus'))
        
        model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
        
        return model
  
    def run_network(self,data=None, model=None, epochs=15, batch= 32, emb_size=256, graph_name='facebook.edges'):
        try:
            
            if data is None:
                X_train, y_train, X_test, y_test = load_data()
            else:
                X_train, y_train, X_test, y_test = data
    
            size=emb_size
            
            if model is None:
                model = self.init_model(size)
                
    
            print( 'Training model...')
    
            history = LossHistory()
            
            t0 = time.time()
            model.fit(X_train, y_train, epochs= epochs, batch_size= 32, callbacks=[history], validation_split= 0.3, verbose=2 ) 
            t_train=time.time()-t0
            
            t1 = time.time()
            preds = model.predict(X_test)
            t_test=time.time()-t1
            
         
        except KeyboardInterrupt:
            print( ' KeyboardInterrupt') 
            
        preds = model.predict(X_test)
    
        pred=[]
        
        f3= open('%s_pred_con_node2vec_%d.txt'%(graph_name, emb_size), 'w') 
        
        for i in range(len(preds)):
            
            pred.append(round(float(preds[i][0])))
            f3.write(str(round(float(preds[i][0]))))
            f3.write('\n')
        f3.close()

     
        rmse=sqrt(mean_squared_error(y_test, pred ))
        mae= mean_absolute_error(y_test, pred )   
              
        return rmse, mae
    

    def short_path(self, G, graph_name, emb_size  ):   
          
        nodes= list(G.nodes())
        edges= list(G.edges())
         
        num_nodes= len(nodes)
        num_edges= len(edges)
        
        ################################## graph #############################
        
        embeddings_file = "./../emb_deepwalk/%s_%d.deepwalk"%(graph_name, emb_size) 
        model = word2vec.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    
     
        ###################################  different k ###################################
        y_train=[]
        X_train=[]
        tr_list1=[]
        tr_list2=[]
        
        f1= open('%s_train.txt'%(graph_name), 'r') 
        for line in f1:
            a=line.strip('\n').split(' ')
             
            tr_list1.append(int(a[0])) 
            tr_list2.append(int(a[1]))
            y_train.append(int(a[2])) 
            
        f1.close()
            
        X_test=[]  
        y_test=[]  
        te_list1=[]
        te_list2=[]    
        f2= open('%s_truth.txt'%(graph_name), 'r') 
        for line in f2:
            a=line.strip('\n').split(' ')
             
            te_list1.append(int(a[0])) 
            te_list2.append(int(a[1]))
            y_test.append(int(a[2]))
        
        f2.close() 
        
        for i in range(len(tr_list1)):
            con= np.concatenate((model[str(tr_list1[i])],model[str(tr_list2[i])]), axis=0)
            X_train.append(con) 
            
        for i in range(len(te_list1)):
            con= np.concatenate((model[str(te_list1[i])],model[str(te_list2[i])]), axis=0)
            X_test.append(con)    
            
        data1= np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        
        RMSE, MAE = self.run_network(data=data1, epochs=15, batch= 32, emb_size= emb_size, graph_name=graph_name)
        
        return  RMSE, MAE , len(X_train), len(X_test) 
                         
            
              
                
            
            
                   
             
                    
                    
           
                
                
                
                
                