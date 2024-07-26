import tensorflow as tf
from keras.optimizers.legacy import Adam
import numpy as np

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

    # The act method takes the current state as input and returns the action to take using the policy network.
    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_network.predict(state)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    # The observe method takes the state, action, reward, next state, and done flag as input 
    # and updates the policy and value networks
    def observe(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.shape[0])
        
        # Calculate advantage
        # The advantage is calculated using the Generalized Advantage Estimation (GAE) method
        value = self.value_network.predict(state)
        next_value = self.value_network.predict(next_state)
        # Measures how much better the taken action was compared to the expected value
        delta = reward + (1 - done) * self.gamma * next_value - value
        advantage = delta + self.gamma * self.lambda_ * (1 - done) * next_value

        with tf.GradientTape() as tape:
            # Ensures the new policy does not deviate significantly from the old policy
            old_action_probs = self.policy_network(state)
            new_action_probs = self.policy_network(state)
            old_log_probs = tf.math.log(tf.reduce_sum(old_action_probs * action_one_hot, axis=1))
            new_log_probs = tf.math.log(tf.reduce_sum(new_action_probs * action_one_hot, axis=1))
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
            
            # Value loss - reduces the prediction error of the value network
            value_loss = tf.reduce_mean(tf.square(reward + (1 - done) * self.gamma * next_value - value))
            loss = policy_loss + 0.5 * value_loss
        
        # Applies the computed gradients to update network parameters
        grads = tape.gradient(loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

    # The save method saves the policy and value networks to the specified paths.
    def save(self, policy_path, value_path):
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
