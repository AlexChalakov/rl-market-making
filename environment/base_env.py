from abc import abstractmethod
import gym
from gym import spaces
import numpy as np

# This base class will provide common functionality that can be inherited and extended by other environment classes in the project.
class BaseMarketEnv(gym.Env):
    def __init__(self, data):
        super(BaseMarketEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = None  # To be defined in child classes
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def calculate_reward(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_current_state(self):
        return self.data.iloc[self.current_step].values

    def advance_step(self):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return done
