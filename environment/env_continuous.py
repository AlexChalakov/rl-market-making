import numpy as np
from environment.base_env import BaseMarketEnv
from gym import spaces

# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 
# This is useful when precise control is needed.

# Market making often requires setting precise bid and ask prices. 
# A continuous action space allows the agent to fine-tune these prices to optimize execution and manage inventory effectively.

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data):
        super(ContinuousMarketEnv, self).__init__(data)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Buy/sell orders

    def step(self, action):
        # Apply action logic here (e.g., update inventory, execute trades)
        done = self.advance_step()
        reward = self.calculate_reward(action)
        state = self.get_current_state()
        return state, reward, done, {}

    def calculate_reward(self, action):
        # Example reward function: reward based on inventory management and execution quality
        reward = -np.abs(action[0])  # Penalize large actions
        return reward
