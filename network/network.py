import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers

# The create_cnn_policy_network function creates a convolutional neural network (CNN) policy network.
# This will determine the actions to take based on the current state.
def create_cnn_attention_policy_network(input_shape):
    l2_reg = regularizers.l2(0.001)
    
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, input_shape=input_shape, kernel_initializer='he_uniform'))
    model.add(layers.Activation(tf.nn.swish))  # Swish activation
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=3, kernel_initializer='he_uniform'))
    model.add(layers.Activation(tf.nn.swish))  # Swish activation
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='swish', kernel_initializer='he_uniform', kernel_regularizer=l2_reg))
    model.add(layers.Dense(128, activation='swish', kernel_initializer='he_uniform', kernel_regularizer=l2_reg))

    # Output layer for actions
    model.add(layers.Dense(2, activation='tanh', kernel_initializer='he_uniform'))
    return model

# The create_cnn_value_network function creates a CNN value network.
# This will estimate the value of the current state.
def create_cnn_attention_value_network(input_shape):
    l2_reg = regularizers.l2(0.001) 
    
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(layers.Activation(tf.nn.swish))  # Swish activation
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=3, kernel_initializer='he_normal'))
    model.add(layers.Activation(tf.nn.swish))  # Swish activation
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='swish', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(layers.Dense(128, activation='swish', kernel_initializer='he_normal', kernel_regularizer=l2_reg))

    # Single output for value estimation
    model.add(layers.Dense(1, activation='linear'))
    return model