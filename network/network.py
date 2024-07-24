# network.py

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

#from tensorflow import keras
#from keras import layers, models

def create_cnn_attention_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Attention()([x, x])
    outputs = layers.Dense(2, activation='tanh')(x)  # Two outputs for buy/sell actions
    model = models.Model(inputs, outputs)
    return model
