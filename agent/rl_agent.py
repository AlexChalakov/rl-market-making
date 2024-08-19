import tensorflow as tf
from keras.optimizers.legacy import Adam
import numpy as np
import pandas as pd

class PPOAgent:
    def __init__(self, env, policy_network, value_network, learning_rate=3e-4, gamma=0.9, clip_range=0.1, epochs=20, batch_size=128, lambda_=0.95, entropy_coeff=0.01):
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
        self.entropy_coeff = entropy_coeff  # Entropy coefficient for encouraging exploration

        # Compile the models with a dummy loss to ensure they are set up correctly
        self.policy_network.compile(optimizer=self.optimizer, loss='mse')
        self.value_network.compile(optimizer=self.optimizer, loss='mse')

    # The act method takes the current state as input and returns the action to take using the policy network.
    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.policy_network.predict(state)[0]
        #print(f"Action output: {action}")
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        #print(f"Clipped action: {action}")
        return action

    # The observe method takes the state, action, reward, next state, and done flag as input 
    # and updates the policy and value networks
    def observe(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        with tf.GradientTape() as tape:
            # Predict action probabilities and state values
            action_pred = self.policy_network(state)
            #print(f"Predicted action before update: {action_pred.numpy()}")  # Debugging line
            value_pred = self.value_network(state)
            next_value_pred = self.value_network(next_state)

            # Calculate advantage
            # The advantage is calculated using the Generalized Advantage Estimation (GAE) method
            delta = reward + (1 - done) * self.gamma * next_value_pred - value_pred
            advantage = delta + self.gamma * self.lambda_ * (1 - done) * next_value_pred

            # PPO objective function: Clipping the ratio to prevent large updates
            old_log_probs = tf.reduce_sum(action_pred * tf.stop_gradient(action), axis=1)
            new_log_probs = tf.reduce_sum(action_pred * action, axis=1)
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

            # Calculate entropy
            # Assuming action_pred are logits, apply softmax if you need probabilities
            # action_probs = tf.nn.softmax(action_pred)  # Uncomment if necessary
            # entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            
            # Since this is a continuous action space, we use the entropy of a Gaussian distribution
            entropy = -0.5 * tf.reduce_sum(1 + tf.math.log(2 * np.pi * tf.clip_by_value(tf.square(action_pred), 1e-10, np.inf)), axis=1)
            entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)
            
            # Value loss for better value function approximation
            value_loss = tf.reduce_mean(tf.square(reward + (1 - done) * self.gamma * next_value_pred - value_pred)) 
            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss

        # Apply the computed gradients to update network parameters
        grads = tape.gradient(loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

        #print(f"Predicted action after update: {action_pred.numpy()}")  # Debugging line

    # The save method saves the policy and value networks to the specified paths.
    def save(self, policy_path, value_path):
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
