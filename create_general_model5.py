#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras import optimizers

from tensorflow.python.framework import ops

from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

from tensorflow.keras import backend as K

def custom_activation(x):

    #return K.sigmoid(x) 

    return K.exp(x)

def create_model():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(325, )),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(100, activation=custom_activation),
        keras.layers.Dense(50, activation=tf.nn.softsign),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])        
    

    return model


if __name__ == "__main__":
        
    model = create_model()
    model.summary()  
