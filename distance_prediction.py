
import numpy as np
import csv
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import networkx as nx

import gensim.models.keyedvectors as word2vec
import keras.callbacks as cb
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def euclidean_distance(vects):
    x, y = vects
    
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1)
    print(shape2)
    return (shape1[0], 1)



def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    print(input_dim)
    seq = Sequential()
    seq.add(Dense(input_dim, input_shape=(input_dim,), activation='softplus'))
    seq.add(Dropout(0.1))
    seq.add(Dense(input_dim, activation='softplus'))
    seq.add(Dropout(0.1))
    seq.add(Dense(input_dim, activation='softplus'))
    return seq



def siamese_net(graph_name='Facebook.edges', emb_size=32, data=None, epochs=20):
     
     
    input_dim = emb_size
    
    tr_pair1, tr_pair2,  y_train, te_pair1, te_pair2, y_test=data
    
    print('X train shape:',tr_pair1.shape)
    print('X test shape:',te_pair1.shape)
    
    print('y train shape:',y_train.shape)
    
    
    # network definition
    base_network = create_base_network(input_dim)
    
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    print(distance)
    
    model = Model(inputs=[input_a, input_b], outputs=distance)
    
    # train
    rms = RMSprop()
    model.compile(loss='mse', optimizer= rms, metrics=['mae'])
    
    history = LossHistory()
    
    model.fit([tr_pair1, tr_pair2], y_train, epochs= epochs, batch_size= 32, callbacks=[history], validation_split= 0.3)
    
    
    preds = model.predict([te_pair1, te_pair2])
    
    pred=[]
    f3= open('./data/%s_prediction_harp_%d.txt'%(graph_name, emb_size), 'w') 
    
    for i in range(len(preds)):
        
        #print(preds[i][0],'  ', y_test[i])
        
        pred.append(round(float(preds[i][0])))
        f3.write(str(preds[i][0]))
        f3.write('\n')
    f3.close()
     
    rmse=sqrt(mean_squared_error(y_test, pred ))
    mae= mean_absolute_error(y_test, pred )
    
    return rmse, mae, y_train.shape[0] ,len(y_test)   




def prepare_data(graph_name, emb):
    
    
    tr_pair_list1=[]
    tr_pair_list2=[]
    y_train=[]
    
    if emb in ['harp', 'poincare']:
    
        model = dict()
        embeddings_file = open( "./data/%s_128.%s"%(graph_name, emb),'r')

        for line in  embeddings_file:
            a=line.strip('\n').split(' ')
            vec=[]
                    
            vec=[float(i) for i in a[1:]]      
            vec1= np.array(vec)    
        
            model.update({a[0]: vec1})
            

    if emb in ['node2vec', 'deepwalk']:        
       
        
        emb_file= './data%s_128.%s'%(graph_name, emb)    
        model = KeyedVectors.load_word2vec_format(emb_file, binary=False)
        
                
       
        
    
    f1= open('./data/%s_train.txt'%(graph_name), 'r') 
    for line in f1:
        a=line.strip('\n').split(' ')
        
        y_train.append(int(a[2])) 
        
        tr_pair_list1.append(model[a[0]]) 
        tr_pair_list2.append(model[a[1]])

    tr_pair1=np.array(tr_pair_list1)
    tr_pair2=np.array(tr_pair_list2)
    y_train= np.array(y_train)
    
    
    te_pair_list1=[]
    te_pair_list2=[]
    y_test=[]
       
    f2= open('./data/%s_test.txt'%(graph_name), 'r') 
    for line in f2:
        at=line.strip('\n').split(' ')
        
        y_test.append(int(at[2])) 
        te_pair_list1.append(model[at[0]]) 
        te_pair_list2.append(model[at[1]])
        
    
    te_pair1=np.array(te_pair_list1)
    te_pair2=np.array(te_pair_list2)
    
    
    return tr_pair1, tr_pair2, y_train, te_pair1, te_pair2, y_test
    
    

if __name__ == '__main__':


    with open('spath.csv', 'w', newline='') as f:
        
        writer = csv.writer(f)
        
        for graph in ['Facebook' ,'Youtube']:
            
            for emb in ['harp']: ##
                
                tr_pair1, tr_pair2,  y_train, te_pair1, te_pair2, y_test= prepare_data(graph, emb)
                
                data= tr_pair1, tr_pair2,  y_train, te_pair1, te_pair2, y_test
                
                rmse, mae, train, test= siamese_net(graph, 128, data, 8)
                
                G=nx. read_edgelist('./data/%s.edges'%graph)
                nodes= len(G.nodes())
                
                
                result=[emb, graph,'nodes', nodes, 'emb_size 128', 'train pairs', train, 'test pairs', test, 'RMSE', rmse,'MAE', mae ]
                print(emb, graph,'nodes', nodes, 'emb_size 128', 'train pairs', train, 'test pairs', test,'RMSE:', rmse,'MAE:', mae)
                writer.writerow(result) 
         
         
    
    
       
