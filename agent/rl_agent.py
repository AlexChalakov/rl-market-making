import tensorflow as tf
from keras.optimizers import Adam
import numpy as np


class PPOAgent:
    def __init__(self, env, policy_network, value_network, learning_rate=3e-5, gamma=0.99, clip_range=0.2, epochs=20, batch_size=256, lambda_=0.95, entropy_coeff=0.01,threshold_depth=200, threshold_volume=100, price_deviation_threshold=0.5):
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network

        # Fine-tune the learning rate, gamma, clip range, epochs, batch size, lambda, and entropy coefficient
        self.optimizer = Adam(learning_rate=learning_rate)
        self.gamma = gamma  # Discount factor
        self.clip_range = clip_range  # Clipping range for PPO - Limits the change in policy updates to improve stability.
        self.epochs = epochs  # Number of epochs for each update
        self.batch_size = batch_size  # Batch size for training
        self.lambda_ = lambda_  # GAE lambda
        self.entropy_coeff = entropy_coeff  # Entropy coefficient for encouraging exploration
        self.losses = []
        self.threshold_depth = threshold_depth
        self.threshold_volume = threshold_volume
        self.price_deviation_threshold = price_deviation_threshold 

        # Compile the models with a dummy loss to ensure they are set up correctly
        self.policy_network.compile(optimizer=self.optimizer, loss='mse')
        self.value_network.compile(optimizer=self.optimizer, loss='mse')

    # The act method takes the current state as input and returns the action to take using the policy network.
    def act(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self.policy_network.predict(state, verbose=0)[0]
        # Extract features from the state
        market_depth = state[0][-5]  # Assuming 3rd element in state array is market_depth
        recent_trade_volume = state[0][-4]  # Assuming 4th element is recent_trade_volume
        recent_trade_price = state[0][-3]  # Assuming last element is recent_trade_price
        mid_price = state[0][-2]  # Assuming 2nd element is mid_price
        
        # Strategy Adjustment based on features
        if market_depth > self.threshold_depth:
            action[1] *= 1.05  # Increase trade size if market depth is high
        if recent_trade_volume > self.threshold_volume:
            action[1] *= 1.2  # Be more aggressive (increase trade size) if recent trade volume is high
        if abs(recent_trade_price - mid_price) > self.price_deviation_threshold:
            action[1] *= 0.9  # Reduce trade size if the recent trade price deviates too much from the mid-price
        
        ofi = state[0][-1]  # Assuming OFI is the last element in state
        
        # Adjust actions based on OFI
        if ofi > 0:  # If there's a buying imbalance
            action[1] *= 1.2  # Encourage selling to balance
        elif ofi < 0:  # If there's a selling imbalance
            action[1] *= 0.9  # Encourage buying to balance
        # Introduce a decay in action size to stabilize inventory and cash
        if np.abs(action[1]) > 0.5:  # Arbitrary threshold for action size
            action[1] *= 0.9  # Decay factor to reduce action size over time
            
        print(f"Action output: {action}")
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        #action = tf.clip_by_value(action, self.env.action_space.low, self.env.action_space.high)
        print(f"Clipped action: {action}")
        return action

    # The observe method takes the state, action, reward, next state, and done flag as input 
    # and updates the policy and value networks
    def observe(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        next_state = np.expand_dims(next_state, axis=0).astype(np.float32)
        
        with tf.GradientTape() as tape:
            # Predict action probabilities and state values
            action_pred = self.policy_network(state)
            #print(f"Action Prediction: {action_pred}")

            value_pred = self.value_network(state)
            next_value_pred = self.value_network(next_state)

            # Calculate advantage using the Generalized Advantage Estimation (GAE) method
            delta = reward + (1 - done) * self.gamma * next_value_pred - value_pred
            advantage = delta + self.gamma * self.lambda_ * (1 - done) * next_value_pred
            # PPO objective function: Clipping the ratio to prevent large updates
            old_log_probs = tf.reduce_sum(action_pred * tf.stop_gradient(action), axis=1)
            new_log_probs = tf.reduce_sum(action_pred * action, axis=1)
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

            # Increase entropy to encourage exploration
            entropy = -0.5 * tf.reduce_sum(1 + tf.math.log(2 * np.pi * tf.clip_by_value(tf.square(action_pred), 1e-10, np.inf)), axis=1)
            entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)
            
            # Value loss for better value function approximation
            value_loss = tf.reduce_mean(tf.square(reward + (1 - done) * self.gamma * next_value_pred - value_pred)) 
            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss

        self.losses.append(loss.numpy())
        # print(f"Total Loss: {loss.numpy()} | Policy Loss: {policy_loss.numpy()} | Value Loss: {value_loss.numpy()} | Entropy Loss: {entropy_loss.numpy()}")

        # Apply the computed gradients to update network parameters
        grads = tape.gradient(loss, self.policy_network.trainable_variables + self.value_network.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in grads]

        self.optimizer.apply_gradients(zip(clipped_grads, self.policy_network.trainable_variables + self.value_network.trainable_variables))

        #print(f"Predicted action after update: {action_pred.numpy()}")  # Debugging line

    # The save method saves the policy and value networks to the specified paths.
    def save(self, policy_path, value_path):
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
        
class TWAPBaseline:
    def __init__(self, env, total_quantity, execution_time):
        self.env = env
        self.total_quantity = total_quantity
        self.execution_time = execution_time
        self.quantity_per_step = total_quantity / execution_time
        self.executed_quantity = 0

    def reset(self):
        """Reset the TWAP strategy."""
        self.executed_quantity = 0

    def act(self, state):
        """Calculate the quantity to trade at each step."""
        # TWAP divides the total quantity equally over the execution time
        if self.executed_quantity < self.total_quantity:
            trade_size = self.quantity_per_step
            self.executed_quantity += trade_size
        else:
            trade_size = 0

        # Randomly decide to buy or sell half the time
        action_type = 1 if np.random.random() > 0.5 else -1
        action = np.array([0, action_type * trade_size])
        
        # Clip the action to the action space of the environment
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return action