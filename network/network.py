import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers

# The create_cnn_policy_network function creates a convolutional neural network (CNN) policy network.
# This will determine the actions to take based on the current state.
def create_cnn_attention_policy_network(input_shape):
    l2_reg = regularizers.l2(0.001)  # Set L2 regularization value (can be tuned)
    
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
    model.add(layers.Dense(2, activation='tanh', kernel_initializer='he_uniform'))  # Output layer for actions (no regularization needed)
    return model


# The create_cnn_value_network function creates a CNN value network.
# This will estimate the value of the current state.
def create_cnn_attention_value_network(input_shape):
    l2_reg = regularizers.l2(0.001)  # Set L2 regularization value (can be tuned)
    
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
    model.add(layers.Dense(1, activation='linear'))  # Single output for value estimation (no regularization needed)
    return model

def create_lstm_policy_network(input_shape, lstm_units=64, output_units=2):
    model = models.Sequential([
        layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal'),
        layers.Dense(output_units, activation='sigmoid', kernel_initializer='GlorotUniform')  # Output layer for actions
    ])
    return model

def create_lstm_value_network(input_shape, lstm_units=64):
    model = models.Sequential([
        layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal'),
        layers.Dense(1, activation='linear')  # Output a continuous value for the state or state-action value
    ])
    return model