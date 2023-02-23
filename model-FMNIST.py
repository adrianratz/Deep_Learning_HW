#!/usr/bin/env python
# coding: utf-8

# # HW 1 
# ## Q1

# In[208]:


#import libraries 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "./tensorflow-datasets/"


# Have to use importlib because files are named with hyphens
import importlib
fmnist_util = importlib.import_module("util-FMNIST")

## Create the model
def create_model(num_hidden_layers, units, dropout_rate, l2_rate, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255)) #normalized within the model
    for i in range(num_hidden_layers):
        model.add(
            tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate)
            )
        )
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

