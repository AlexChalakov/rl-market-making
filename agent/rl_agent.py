import tensorflow as tf
from keras.optimizers.legacy import Adam
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from configurations import LOGGER
import os

class PPOAgent:
    def __init__(self, env, policy_network, value_network, learning_rate=1e-4, gamma=0.99, clip_range=0.2, epochs=10, batch_size=64, lambda_=0.95):
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network

        # Fine-tune the learning rate, gamma, clip range, epochs, batch size, and lambda
        self.optimizer = Adam(learning_rate=learning_rate)
        self.gamma = gamma  # Discount factor
        self.clip_range = clip_range  # Clipping range for PPO - Limits the change in policy updates to improve stability.
        self.epochs = epochs  # Number of epochs for each update
        self.batch_size = batch_size  # Batch size for training
        self.lambda_ = lambda_  # GAE lambda

        # Compile the models with a dummy loss to ensure they are set up correctly
        self.policy_network.compile(optimizer=self.optimizer, loss='mse')
        self.value_network.compile(optimizer=self.optimizer, loss='mse')

    # The act method takes the current state as input and returns the action to take using the policy network.
    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.policy_network.predict(state)[0]
        return action

    # The observe method takes the state, action, reward, next state, and done flag as input 
    # and updates the policy and value networks
    def observe(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        with tf.GradientTape() as tape:
            action_pred = self.policy_network(state)
            value_pred = self.value_network(state)
            next_value_pred = self.value_network(next_state)

            # Calculate advantage
            # The advantage is calculated using the Generalized Advantage Estimation (GAE) method
            delta = reward + (1 - done) * self.gamma * next_value_pred - value_pred
            advantage = delta + self.gamma * self.lambda_ * (1 - done) * next_value_pred

            # Ensures the new policy does not deviate significantly from the old policy - PPO loss
            old_log_probs = tf.reduce_sum(action_pred * tf.stop_gradient(action), axis=1)
            new_log_probs = tf.reduce_sum(action_pred * action, axis=1)
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
            
            # Value loss - reduces the prediction error of the value network
            value_loss = tf.reduce_mean(tf.square(reward + (1 - done) * self.gamma * next_value_pred - value_pred))
            loss = policy_loss + 0.5 * value_loss

        # Applies the computed gradients to update network parameters
        grads = tape.gradient(loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

    # The save method saves the policy and value networks to the specified paths.
    def save(self, policy_path, value_path):
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
        
class QLearningAgent:
    def __init__(self, env, state_size, action_size, bins=10, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.bins = bins
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))
        
    
    #  The act method selects an random action if its smailler than epsilon or it return the action with the highest Q-value for the current state 
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[discretized_state])
    
    # The observe method takes the state, action, reward, next state, and done flag as input 
    def observe(self, state, action, reward, next_state, done):
        q_update = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        # Updates the Q-value for the current state-action pair using the learning rate.
        self.q_table[state, action] += self.lr * (q_update - self.q_table[state, action])
        
