import numpy as np
from environment.base_env import BaseMarketEnv
from gym import spaces

# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data):
        super(ContinuousMarketEnv, self).__init__(data)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Define action space for bid and ask price adjustments
        self.inventory = 0 # Inventory for the agent
        self.cash = 0 # Cash for the agent
        self.trades = [] # List to store executed trades

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self):
        self.current_step = 0
        self.inventory = 10
        self.cash = 10000
        self.trades = []
        return self.data.iloc[self.current_step].values

    # The step method takes an action as input and returns the next state, reward, done flag, and additional information.
    # Simulates order executions. 
    # If the bid adjustment is enough to buy at the best bid price, it buys; 
    # if the ask adjustment is enough to sell at the best ask price, it sells.
    def step(self, action):
        bid_adjustment, ask_adjustment = action
        best_bid = self.data.iloc[self.current_step]['Bid Price 1']
        best_ask = self.data.iloc[self.current_step]['Ask Price 1']
        spread = best_ask - best_bid

        # Normalize adjustments by the spread size
        bid_adjustment = bid_adjustment * spread / 2
        ask_adjustment = ask_adjustment * spread / 2

        # Simulate market order execution
        # best_bid is highest bid price available, best_ask is lowest ask price available
        # bid_adjustment and ask_adjustment are the adjustments to the bid and ask prices, how much the agent should adjust...
        # executed_bid is the price at which the agent buys, executed_ask is the price at which the agent sells
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask + ask_adjustment

        # Buying action (if there's enough cash and the bid adjustment leads to a valid buy)
        if executed_bid <= self.cash:
            self.inventory += 1
            self.cash -= executed_bid
            self.trades.append(("BUY", executed_bid))
            action_taken = True
            print(f"BUY: {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

        # Selling action (if there's enough inventory and the ask adjustment leads to a valid sell)
        if self.inventory > 0:
            self.inventory -= 1
            self.cash += executed_ask
            self.trades.append(("SELL", executed_ask))
            action_taken = True
            print(f"SELL: {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

        done = self.advance_step()
        reward = self.calculate_reward()

        # Add a small positive reward for taking an action
        if action_taken:
            reward += 0.01  # Small reward for making a trade

        state = self.get_current_state()
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self):
        pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1']

        # Apply a scaling factor to reduce the overall reward magnitude
        scaling_factor = 0.001  # Adjust this factor as needed to scale down rewards

        # Penalize if inventory is zero (to encourage maintaining inventory)
        inventory_penalty = 0
        if self.inventory == 0:
            inventory_penalty = 1000 * scaling_factor # Significant penalty for having zero inventory

        # Reward for maintaining some inventory
        balance_reward = max(self.inventory, 0) * 0.1 * scaling_factor # Small reward for maintaining inventory

        # Cash management penalty (discourage negative cash) but less severe
        cash_penalty = max(0, -self.cash) * 0.005 * scaling_factor # Reduced penalty for negative cash

        # Execution quality reward
        executed_prices = [trade[1] for trade in self.trades]
        if executed_prices:
            average_execution_price = np.mean(executed_prices)
            mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
            execution_quality_reward = -abs(average_execution_price - mid_price) * 0.02 * scaling_factor # Reduced impact of execution quality
        else:
            execution_quality_reward = 0

        # Total reward
        reward = (pnl - inventory_penalty - cash_penalty + execution_quality_reward + balance_reward) * scaling_factor
        return reward
