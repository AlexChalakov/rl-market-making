import tensorflow as tf
from keras.optimizers.legacy import Adam
import numpy as np

class PPOAgent:
    def __init__(self, env, network):
        self.env = env
        self.network = network
        self.optimizer = Adam(learning_rate=1e-3)
        self.gamma = 0.99  # Discount factor

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.network.predict(state)[0]
        return action

    def observe(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        with tf.GradientTape() as tape:
            q_value = self.network(state)
            next_q_value = self.network(next_state)
            target = reward + (1 - done) * self.gamma * next_q_value
            loss = tf.reduce_mean(tf.square(target - q_value))
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

    def save(self, path):
        self.network.save(path)
