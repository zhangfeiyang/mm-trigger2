#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers


def create_model():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(17739, )),
        #keras.layers.Dense(2),
        #keras.layers.Dense(2, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])        
    
    #model = keras.Sequential()
    #model.add(keras.layers.Embedding(17739, 50))
    #model.add(keras.layers.GlobalAveragePooling1D())
    #model.add(keras.layers.Dense(50, activation=tf.nn.relu))
    #model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    return model


if __name__ == "__main__":
        
    model = create_model()
    model.summary()  
