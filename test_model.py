#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

import create_easy_model
import create_general_model
import gen_dataset
import gen_dataset3
import gen_dataset_test

#model = create_easy_model.create_model()
model = create_general_model.create_model()
import os
datas = gen_dataset_test.gen_data(1)
data = datas[0]
label = datas[1]

model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
model.load_weights('./checkpoints/my_checkpoint')
result = model.predict(data)
N = len(datas[0])

file_result = open('result','w')

for i in range(N):
    file_result.write(str(result[i][0])+'\t'+str(result[i][1])+'\t'+str(label[i][0])+'\t'+str(datas[2])+'\n')

file_result.close()

