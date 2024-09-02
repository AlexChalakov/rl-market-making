import numpy as np
from environment.base_env import BaseMarketEnv
from gym import spaces
# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data, reward_type='default'):
        super(ContinuousMarketEnv, self).__init__(data)
        # Adjust action space to include trade size (units) in addition to price adjustments
        self.action_space = spaces.Box(low=np.array([-1, -5]), high=np.array([1, 5]), dtype=np.float32)
        self.inventory = 0 # Inventory for the agent
        self.cash = 10 # Cash for the agent
        self.trades = [] # List to store executed trades
        self.reward_type = reward_type
        self.past_pnls = []  # To store past PnL values for risk metrics
        self.trade_flows = []  # To store order flow imbalance data
        self.current_step = 0
        self.action_count = 0

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self):
        self.current_step = np.random.randint(0, 10)  # Start from a random step within the first 10 steps
        self.inventory = 0
        self.cash = 9 + np.random.uniform(-1, 1)  # Add a random variation to the cash
        self.trades = []
        self.past_pnls = []
        self.trade_flows = []
        self.action_count = 0
        return self.data.iloc[self.current_step].values

    # The step method takes an action as input and returns the next state, reward, done flag, and additional information.
    # Simulates order executions. 
    # If the bid adjustment is enough to buy at the best bid price, it buys; 
    # if the ask adjustment is enough to sell at the best ask price, it sells.
    def step(self, action):
        bid_adjustment, trade_size = action
        best_bid = self.data.iloc[self.current_step]['BidPrice_0']
        best_ask = self.data.iloc[self.current_step]['AskPrice_0']
        spread = best_ask - best_bid

        # Enhanced action-to-order mapping logic
        bid_adjustment = bid_adjustment * spread
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask - bid_adjustment  # Symmetrical adjustment

        action_taken = False

        if trade_size != 0:  # Ensure trade size is not zero
            # Buy action (if cash is sufficient)
            if trade_size > 0 and self.cash >= executed_bid * trade_size:
                self.inventory += trade_size
                self.cash -= executed_bid * trade_size
                self.trades.append(("BUY", executed_bid, trade_size))
                action_taken = True
                print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Sell action (if inventory is sufficient)
            if trade_size < 0 and self.inventory >= abs(trade_size):
                self.inventory += trade_size  # trade_size is negative, so this decreases inventory
                self.cash += abs(trade_size) * executed_ask
                self.trades.append(("SELL", executed_ask, abs(trade_size)))
                action_taken = True
                print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Increment the action counter when an action is taken
            self.action_count += 1

        done = self.advance_step()
        # Calculate reward based on the action taken
        reward = self.calculate_reward(action_taken)

        state = self.get_current_state()
        
        # Store PnL and trade flow data
        self.past_pnls.append(self.cash + self.inventory * self.data.iloc[self.current_step]['BidPrice_0'])
        self.trade_flows.append(self.cash) 
        
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self, action_taken):
        # Calculate the percentage of half spread to the current bid price
        #half_spread_percentage = (self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']) / self.data.iloc[self.current_step]['BidPrice_0']
        # Calculate the Profit and Loss (PnL)
        pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['BidPrice_0']
        
        # New metrics calculation
        #implementation_shortfall = self.implementation_shortfall()
        #order_flow_imbalance = self.order_flow_imbalance()
        #rsi = self.rsi()
        #mean_average_pricing = self.mean_average_pricing()

        if self.reward_type == 'default':
            reward = self._default_reward(pnl, action_taken)
        elif self.reward_type == 'asymmetrical':
            reward = self._asymmetrical_reward(pnl, action_taken)
        elif self.reward_type == 'realized_pnl':
            reward = self._realized_pnl_reward(pnl)
        elif self.reward_type == 'trade_completion':
            reward = self._trade_completion_reward(pnl, action_taken)
        else:
            reward = self._default_reward(pnl, action_taken)  # Fallback to default

        return reward

    # The _default_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
    # This is the main reward function used in the environment.
    def _default_reward(self, pnl, action_taken):
        # Reward for making a trade
        trade_reward = 1.5 if action_taken else 0  # Encourages action

        # Inventory penalty to encourage balanced inventory
        target_inventory = 5  # Adjusted target inventory for simplicity
        dynamic_inventory_penalty = 0.5 if abs(self.inventory - target_inventory) > 2 else 0.07 # Dynamic penalty based on distance from target
        inventory_penalty = max(0, abs(self.inventory - target_inventory)) * dynamic_inventory_penalty

        # Execution quality reward/penalty based on how close the executed price is to the mid-market price
        mid_price = (self.data.iloc[self.current_step]['BidPrice_0'] + self.data.iloc[self.current_step]['AskPrice_0']) / 2
        spread = self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']
        execution_quality_reward = 0
        if self.trades:
            # Get the last trade details
            last_trade = self.trades[-1]
            # Unpack the trade details
            trade_type, executed_price, trade_size = last_trade

            # Penalize execution quality if executed price deviates significantly from mid price
            if abs(executed_price - mid_price) > 0.5 * spread:
                # Penalize more for larger deviations
                execution_quality_reward = -abs(executed_price - mid_price) * 0.12
            else:
                # Reward for good execution quality
                execution_quality_reward = -abs(executed_price - mid_price) * 0.06

        # Spread capture reward to encourage effective spread capture
        spread_capture_reward = 0
        if self.trades:
            if trade_type == "BUY":
                spread_capture_reward = (mid_price - executed_price) * 0.07
            elif trade_type == "SELL":
                spread_capture_reward = (executed_price - mid_price) * 0.07

        # Small reward for maintaining or increasing PnL over time
        pnl_change_reward = (pnl - self.past_pnls[-1]) * 0.02 if len(self.past_pnls) > 1 else 0

        # Adding the market metrics to the reward
        # These components are meant to encourage the agent to optimize for better market conditions
        #spread_penalty = -abs(half_spread_percentage) * 0.0005
        #shortfall_penalty = -abs(implementation_shortfall) * 0.0005
        #imbalance_reward = -abs(order_flow_imbalance) * 0.0005  # You might want to consider reversing this sign depending on your interpretation
        #rsi_penalty = -abs(rsi - 50) * 0.0001  # Penalize deviation from a neutral RSI
        #map_reward = -abs(mean_average_pricing - mid_price) * 0.0005

        # Total reward is the sum of the components
        reward = (
            pnl 
            + trade_reward 
            - inventory_penalty 
            + execution_quality_reward 
            + spread_capture_reward 
            + pnl_change_reward 
            #+ spread_penalty 
            #+ shortfall_penalty 
            #+ imbalance_reward 
            #+ rsi_penalty 
            #+ map_reward
        )

        return reward

    # The _asymmetrical_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
    def _asymmetrical_reward(self, pnl, action_taken):
        # Similar to default, but penalize negative pnl more harshly and reward positive pnl slightly more
        reward = self._default_reward(pnl, action_taken)

        if pnl < 0:
            reward *= 1.5  # Increase penalty for negative pnl
        elif pnl > 0:
            reward *= 1.1  # Increase reward for positive pnl

        return reward

    # The _realized_pnl_reward method focuses purely on the realized PnL from the trades.
    def _realized_pnl_reward(self):
        # Focuses purely on the realized PnL from the trades
        realized_pnl = sum([trade[1] * trade[2] for trade in self.trades if trade[0] == "SELL"])
        return realized_pnl * 0.01

    # The _trade_completion_reward method rewards completing trades and penalizes incomplete trades or open positions.
    def _trade_completion_reward(self, action_taken):
        # Rewards completing trades and penalizes incomplete trades or open positions
        reward = 0.0

        if action_taken:
            reward += 0.02  # Reward for completing a trade

        if self.inventory == 0:
            reward += 0.05  # Additional reward for balancing inventory
        else:
            reward -= 0.01 * abs(self.inventory)  # Penalize for holding inventory

        return reward

    # The _spread_capture_reward method focuses on capturing the spread between the bid and ask prices.
    def implementation_shortfall(self):
        # Implementation Shortfall (IS) calculation
        if not self.trades:
            return 0
        mid_price = (self.data.iloc[self.current_step]['BidPrice_0'] + self.data.iloc[self.current_step]['AskPrice_0']) / 2
        total_is = sum(abs(trade[1] - mid_price) * trade[2] for trade in self.trades)
        return total_is / len(self.trades) if self.trades else 0

    # The order_flow_imbalance method calculates the Order Flow Imbalance (OFI) based on the trade flows.
    def order_flow_imbalance(self):
        # Order Flow Imbalance (TFI) calculation
        if not self.trade_flows:
            return 0
        total_buy = sum(flow for flow in self.trade_flows if flow > 0)
        total_sell = sum(flow for flow in self.trade_flows if flow < 0)
        return total_buy - total_sell
    
    # The rsi method calculates the Relative Strength Index (RSI) based on the executed sell prices.
    def rsi(self):
        # RSI calculation (simple version)
        prices = [trade[1] for trade in self.trades if trade[0] == "SELL"]
        if len(prices) < 14:
            return 50  # Default RSI value
        gains = [prices[i] - prices[i - 1] for i in range(1, len(prices)) if prices[i] > prices[i - 1]]
        losses = [-1 * (prices[i] - prices[i - 1]) for i in range(1, len(prices)) if prices[i] < prices[i - 1]]
        average_gain = np.mean(gains) if gains else 0
        average_loss = np.mean(losses) if losses else 0
        rs = average_gain / average_loss if average_loss != 0 else float('inf')
        return 100 - (100 / (1 + rs))

    # The mean_average_pricing method calculates the Mean Average Pricing (MAP) based on the executed trades.
    def mean_average_pricing(self):
        # Mean Average Pricing (MAP) calculation
        if not self.trades:
            return 0
        total_price = sum(trade[1] * trade[2] for trade in self.trades)
        total_quantity = sum(trade[2] for trade in self.trades)
        return total_price / total_quantity if total_quantity else 0