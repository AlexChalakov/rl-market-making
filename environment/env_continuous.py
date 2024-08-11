import numpy as np
from base_env import BaseMarketEnv
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
        self.inventory = 0
        self.cash = 0
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

        # Simulate market order execution
        # best_bid is highest bid price available, best_ask is lowest ask price available
        # bid_adjustment and ask_adjustment are the adjustments to the bid and ask prices, how much the agent should adjust...
        # executed_bid is the price at which the agent buys, executed_ask is the price at which the agent sells
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask + ask_adjustment

        if executed_bid >= best_bid:
            self.inventory += 1
            self.cash -= executed_bid
            self.trades.append(("BUY", executed_bid))
            print(f"BUY: {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")
        if executed_ask <= best_ask:
            self.inventory -= 1
            self.cash += executed_ask
            self.trades.append(("SELL", executed_ask))
            print(f"SELL: {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

        # Calculate reward based on the action taken
        done = self.advance_step()
        reward = self.calculate_reward()
        state = self.get_current_state()
        #print(f"Next state: {state}, Reward: {reward}, Done: {done}")
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self):
        # the total value of the agent's holdings is calculated as the sum of cash and inventory multiplied by the best bid price.
        # Calculate the Profit and Loss (PnL) based on the current inventory and cash position.
        pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1']

        # Penalize the agent for holding excess inventory by calculating the inventory risk.
        inventory_risk = abs(self.inventory) * 0.1

        # Calculate the execution quality
        # We judge it by comparing it to the mid price, which is the average of the best bid and ask prices.
        executed_prices = [trade[1] for trade in self.trades]   # Extract executed prices from trade history
        if executed_prices:
            average_execution_price = np.mean(executed_prices)  # Calculate average execution price
            mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2 # Calculate mid price
            execution_quality_reward = -abs(average_execution_price - mid_price) * 0.1 # Penalize deviation from mid price / the closer the execution price to the mid price, the higher the reward
        else:
            execution_quality_reward = 0

        # The reward is a combination of the PnL, inventory risk, and execution quality components.
        reward = pnl - inventory_risk + execution_quality_reward
        return reward
