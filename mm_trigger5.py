#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

import create_general_model5
#import gen_dataset5

NN = 15 
N2 = 100
#model = create_easy_model.create_model()
model = create_general_model5.create_model()
import os

filename = 'Data5_'+str(NN)
if os.path.exists(filename):
    
    file0 = open(filename,'r')
    lines = file0.readlines()
    N = len(lines)

    train_data  =[]
    train_label =[]
    test_data   =[]
    test_label  =[]
    
    for i in range(0,int(N*3/4)):
        datas = lines[i].split()
        data = []
        for j in range(NN*NN + N2):
            data.append(int(datas[j]))
        train_data.append(data)
        try: 
            train_label.append([int(datas[-1])])
        except IndexError:
            print(i)
        
    for i in range(int(N*3/4),N):
        datas = lines[i].split()
        data = []
        for j in range(NN*NN + N2):
            data.append(int(datas[j]))
        test_data.append(data)
        try: 
            test_label.append([int(datas[-1])])
        except IndexError:
            print(i)

        
    train_data  = np.array(train_data,'i') 
    train_label = np.array(train_label,'i')
    test_data   = np.array(test_data,'i')
    test_label  = np.array(test_label,'i')


#else:
#
#    train_datas = gen_dataset5.gen_data(0)
#    train_data  = train_datas[0]
#    train_label = train_datas[1]
#    
#    test_datas = gen_dataset5.gen_data(1)
#    test_data =  test_datas[0]
#    test_label = test_datas[1]


#model.compile(optimizer=tf.train.AdamOptimizer(),
#model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate = 0.002),
model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate = 0.0005,momentum=0.3),
                      loss='sparse_categorical_crossentropy',
                      #loss='binary_crossentropy',
                      metrics=['accuracy'])

model.fit(
        train_data,
        train_label,
        epochs=200,
        #batch_size=10,
        validation_data = (test_data,test_label),
        verbose = 1
        )
model.save_weights('./checkpoints/my_checkpoint')

test_loss, test_acc = model.evaluate(test_data,test_label)
print('Test accuracy:', test_acc)

result = model.predict(test_data)
N = len(result)
file_result = open('result','w')

for i in range(N):
        file_result.write(str(result[i][0])+'\t'+str(result[i][1])+'\t'+str(test_label[i][0])+'\n')
file_result.close()

