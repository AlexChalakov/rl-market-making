import numpy as np
from environment.base_env import BaseMarketEnv
from gym import spaces

# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data):
        super(ContinuousMarketEnv, self).__init__(data)
        # Adjust action space to include trade size (units) in addition to price adjustments
        self.action_space = spaces.Box(low=np.array([-1, -5]), high=np.array([1, 5]), dtype=np.float32)  # Trade size between -5 and 5 units
        self.inventory = 0 # Inventory for the agent
        self.cash = 0 # Cash for the agent
        self.trades = [] # List to store executed trades

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self):
        self.current_step = 0
        self.inventory = 0
        self.cash = 10000
        self.trades = []
        return self.data.iloc[self.current_step].values

    # The step method takes an action as input and returns the next state, reward, done flag, and additional information.
    # Simulates order executions. 
    # If the bid adjustment is enough to buy at the best bid price, it buys; 
    # if the ask adjustment is enough to sell at the best ask price, it sells.
    def step(self, action):
        bid_adjustment, trade_size = action
        best_bid = self.data.iloc[self.current_step]['Bid Price 1']
        best_ask = self.data.iloc[self.current_step]['Ask Price 1']
        spread = best_ask - best_bid

        # Adjust bid based on spread
        bid_adjustment = bid_adjustment * spread  # Remove the /2 to increase the adjustment range
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask + bid_adjustment

        action_taken = False
        trade_size = int(trade_size)  # Convert trade size to integer

        if trade_size != 0:  # Ensure trade size is not zero
            # Buying action (if there's enough cash and the bid adjustment leads to a valid buy)
            if trade_size > 0 and self.cash >= executed_bid * trade_size:
                self.inventory += trade_size
                self.cash -= executed_bid * trade_size
                self.trades.append(("BUY", executed_bid, trade_size))
                action_taken = True
                print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Selling action (if there's enough inventory and the ask adjustment leads to a valid sell)
            if trade_size < 0 and self.inventory >= abs(trade_size):
                self.inventory += trade_size  # trade_size is negative, so this decreases inventory
                self.cash += abs(trade_size) * executed_ask
                self.trades.append(("SELL", executed_ask, abs(trade_size)))
                action_taken = True
                print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

        done = self.advance_step()
        reward = self.calculate_reward()

        # Add a small positive reward for taking an action
        if action_taken:
            reward += 0.01  # Small reward for making a trade

        state = self.get_current_state()
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self):
        # Calculate the Profit and Loss (PnL)
        pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1']

        # Apply a scaling factor to reduce the overall reward magnitude
        scaling_factor = 0.01

        # Inventory penalty to encourage balanced inventory (not too high or too low)
        inventory_penalty = max(0, abs(self.inventory - 5)) * 0.05  # Adjusted to be less harsh

        # Reward for maintaining some inventory, penalize for zero inventory
        balance_reward = max(self.inventory, 0) * 0.1  # Increased reward for holding inventory

        # Cash management penalty (discourage negative cash) but less severe
        cash_penalty = max(0, -self.cash) * 0.01  # Slightly reduced penalty for negative cash

        # Execution quality reward
        executed_prices = [trade[1] for trade in self.trades]
        if executed_prices:
            average_execution_price = np.mean(executed_prices)
            mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
            execution_quality_reward = -abs(average_execution_price - mid_price) * 0.01  # Adjusted to be less harsh
        else:
            execution_quality_reward = 0

        # Total reward
        reward = (pnl - inventory_penalty - cash_penalty + execution_quality_reward + balance_reward) * scaling_factor

        return reward
