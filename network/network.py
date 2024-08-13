import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import regularizers

# The create_cnn_policy_network function creates a convolutional neural network (CNN) policy network.
# This will determine the actions to take based on the current state.
def create_cnn_attention_policy_network(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Attention()([x, x])
    outputs = layers.Dense(2, activation='tanh')(x)  # Two outputs for buy/sell actions
    outputs = layers.Lambda(lambda x: x * 0.5)(outputs)  # Scale output to [-0.5, 0.5] to reduce extreme actions
    model = models.Model(inputs, outputs)
    return model

# The create_cnn_value_network function creates a CNN value network.
# This will estimate the value of the current state.
def create_cnn_attention_value_network(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)  # Add dropout to prevent overfitting
    x = layers.Attention()([x, x])
    outputs = layers.Dense(1, activation='linear')(x)  # Single output for value estimation
    model = models.Model(inputs, outputs)
    return model
