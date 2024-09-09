# import numpy as np
# from environment.base_env import BaseMarketEnv
# from gymnasium import spaces
# import time
# # In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# # The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
# ## This code simulates a market making environment with a focus on inventory management and execution quality.
# class ContinuousMarketEnv(BaseMarketEnv):
#     def __init__(self, data, reward_type='default'):
#         super(ContinuousMarketEnv, self).__init__(data)
#         # Adjust action space to include trade size (units) in addition to price adjustments
#         self.action_space = spaces.Box(low=np.array([-1, -10]), high=np.array([1, 10]), dtype=np.float32)
#         self.observation_space = spaces.Box(low=0, high=1, shape=(94,), dtype=np.float32)
#         self.inventory = 0 # Inventory for the agent
#         self.cash = 10000 # Cash for the agent
#         self.trades = [] # List to store executed trades
#         self.reward_type = reward_type
#         self.past_pnls = []  # To store past PnL values for risk metrics
#         self.trade_flows = []  # To store order flow imbalance data
#         self.current_step = 0
#         self.action_count = 0

#     # The reset method initializes the environment at the beginning of an episode.
#     # It resets the current step, inventory, cash, and trade history.
#     def reset(self, seed=None):
#         if seed is None:
#             seed = int(time.time())  # Use current time as the seed
#         self.current_step = np.random.randint(0, len(self.data)-501)  # Start from a random step within the first 10 steps
#         self.inventory = 0
#         self.cash = 10000
#         self.trades = []
#         self.past_pnls = []
#         self.trade_flows = []
#         self.action_count = 0
#         return self.data.iloc[self.current_step].values
#         # return np.concatenate([self.data.iloc[self.current_step].values, np.zeros(7)]) , None

#     # The step method takes an action as input and returns the next state, reward, done flag, and additional information.
#     # Simulates order executions. 
#     # If the bid adjustment is enough to buy at the best bid price, it buys; 
#     # if the ask adjustment is enough to sell at the best ask price, it sells.
    
#     # def get_current_state(self):
#     #     base_state = self.data.iloc[self.current_step].values

#     #     # Calculate additional features
#     #     # market_depth = self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']
#     #     # recent_trade_volume = sum([trade[2] for trade in self.trades[-5:]])  # Sum of last 5 trades
#     #     # recent_trade_price = self.trades[-1][1] if self.trades else 0  # Price of last trade
#     #     # mid_price = (self.data.iloc[self.current_step]['Ask Price 1'] + self.data.iloc[self.current_step]['Bid Price 1']) / 2
        
#     #     # for crypto data
#     #     market_depth = self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']
#     #     recent_trade_volume = sum([trade[2] for trade in self.trades[-5:]])  # Sum of last 5 trades
#     #     recent_trade_price = self.trades[-1][1] if self.trades else 0  # Price of last trade
#     #     mid_price = (self.data.iloc[self.current_step]['AskPrice_0'] + self.data.iloc[self.current_step]['BidPrice_0']) / 2
#     #     ofi = self.order_flow_imbalance()

#     #     # Combine base state with additional features
#     #     full_state = np.concatenate([base_state, [self.inventory, self.cash, market_depth, recent_trade_volume, recent_trade_price, mid_price, ofi]])
#     #     return full_state
    
#     def step(self, action):
#         bid_adjustment, trade_size = action
        
#          # Action refinement: scaling trade_size down if conditions met
#         if self.calculate_sharpe_ratio() < 1.0:
#             trade_size *= 0.5  # Reduce action size for conservative play
            
#         # best_bid = self.data.iloc[self.current_step]['Bid Price 1']
#         # best_ask = self.data.iloc[self.current_step]['Ask Price 1']
        
#         # for crypto data
#         best_bid = self.data.iloc[self.current_step]['BidPrice_0']
#         best_ask = self.data.iloc[self.current_step]['AskPrice_0']
        
#         spread = best_ask - best_bid
        
#         # Enhanced action-to-order mapping logic
#         bid_adjustment = bid_adjustment * spread
#         executed_bid = best_bid + bid_adjustment
#         executed_ask = best_ask - bid_adjustment  # Symmetrical adjustment

#         action_taken = False

#         if trade_size != 0:  # Ensure trade size is not zero
#             # Buy action (if cash is sufficient)
#             if trade_size > 0 and self.cash >= executed_bid * trade_size:
#                 self.inventory += trade_size
#                 self.cash -= executed_bid * trade_size
#                 self.trades.append(("BUY", executed_bid, trade_size))
#                 action_taken = True
#                 # print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

#             # Sell action (if inventory is sufficient)
#             if trade_size < 0 and self.inventory >= abs(trade_size):
#                 self.inventory += trade_size  # trade_size is negative, so this decreases inventory
#                 self.cash += abs(trade_size) * executed_ask
#                 self.trades.append(("SELL", executed_ask, abs(trade_size)))
#                 action_taken = True
#                 # print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

#             # Increment the action counter when an action is taken
#             self.action_count += 1
            
#         # Calculate additional metrics for logging and analysis
#         sharpe_ratio = self.calculate_sharpe_ratio()
#         map_value = self.calculate_mean_absolute_position()
#         mdd_value = self.calculate_maximum_drawdown()
#         done = self.advance_step()
        
#         # Calculate reward based on the action taken
#         reward = self.calculate_reward(action_taken)
#         state = self.get_current_state()

#         # # Calculate the Profit and Loss (PnL)
#         # # Realized PnL: Profit from closed trades
#         # realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])

#         # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
#         # unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
#         # Store PnL and trade flow data
#         # self.past_pnls.append((self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1'])/10000)
        
#         #for crypto data
#         realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])
#         # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
#         unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['BidPrice_0']

#         # Total PnL
#         pnl = (self.cash + realized_pnl + unrealized_pnl) / 10000
#         self.past_pnls.append(pnl)
#         self.trade_flows.append(self.cash) 
        
#         # half_spread_percentage = self.half_spread_percentage()
#         # implementation_shortfall = self.implementation_shortfall()
#         # order_flow_imbalance = self.order_flow_imbalance()
#         # rsi = self.rsi()
        
#         # Log metrics for debugging and performance analysis
#         # print(f"Step: {self.current_step}, Sharpe Ratio: {sharpe_ratio}, MAP: {map_value}, MDD: {mdd_value}")
        
#         return state, reward, done, None ,{}
        
#     # Designed to balance the inventory and execution quality.
#     def calculate_reward(self, action_taken):
#         # Calculate the percentage of half spread to the current bid price
#         half_spread_percentage = self.half_spread_percentage()
#         # # Calculate the Profit and Loss (PnL)
#         # # Realized PnL: Profit from closed trades
#         # realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])

#         # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
#         # unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
        
#         # for crypto data
#         # Calculate the Profit and Loss (PnL)
#         # Realized PnL: Profit from closed trades
#         realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])
#         # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
#         unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['BidPrice_0']

#         # Total PnL
#         pnl = (self.cash + realized_pnl + unrealized_pnl) / 10000  # Normalize by dividing by initial cash
        
#         # New metrics calculation
#         implementation_shortfall = self.implementation_shortfall()
#         map_value = self.calculate_mean_absolute_position()
#         mdd_value = self.calculate_maximum_drawdown()
#         sharpe_ratio = self.calculate_sharpe_ratio()

#         if self.reward_type == 'default':
#             reward = self._default_reward(pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio)
#         elif self.reward_type == 'asymmetrical':
#             reward = self._asymmetrical_reward(pnl, action_taken)
#         elif self.reward_type == 'realized_pnl':
#             reward = self._realized_pnl_reward(pnl)
#         elif self.reward_type == 'trade_completion':
#             reward = self._trade_completion_reward(pnl, action_taken)
#         elif self.reward_type == 'spread_capture':
#             reward = self._spread_capture_reward(pnl, action_taken)
#         else:
#             reward = self._default_reward(pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio)  # Fallback to default

#         return reward

#     # The _default_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
#     # This is the main reward function used in the environment.
#     def _default_reward(self, pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio):
#         window_size = 5
#         # Reward for making a trade
#         trade_reward = 0.5 if action_taken else 0  # Encourages action

#         # Inventory penalty to encourage balanced inventory
#         target_inventory = 1  # Adjusted target inventory for simplicity
#         dynamic_inventory_penalty = 0.1 if abs(self.inventory - target_inventory) > 2 else 0.01 # Dynamic penalty based on distance from target
#         inventory_penalty = max(0, abs(self.inventory - target_inventory)) * dynamic_inventory_penalty

#         # # Execution quality reward/penalty based on how close the executed price is to the mid-market price
#         # mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
#         # spread = self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']
#         # for crypto data
#         mid_price = (self.data.iloc[self.current_step]['BidPrice_0'] + self.data.iloc[self.current_step]['AskPrice_0']) / 2
#         spread = self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']
        
#         execution_quality_reward = 0
#         if self.trades:
#             # Get the last trade details
#             last_trade = self.trades[-1]
#             # Unpack the trade details
#             trade_type, executed_price, trade_size = last_trade

#             # Penalize execution quality if executed price deviates significantly from mid price
#             if abs(executed_price - mid_price) > 0.5 * spread:
#                 # Penalize more for larger deviations
#                 execution_quality_reward = -abs(executed_price - mid_price) * 0.1
#             else:
#                 # Reward for good execution quality
#                 execution_quality_reward = -abs(executed_price - mid_price) * 0.05

#         # Calculate rolling average cash and inventory
#         if len(self.trade_flows) > window_size:
#             rolling_avg_cash = np.mean(self.trade_flows[-window_size:])
#         else:
#             rolling_avg_cash = self.cash  # If not enough data, use current cash

#         if len(self.trades) > window_size:
#             rolling_avg_inventory = np.mean([trade[2] for trade in self.trades[-window_size:]])
#         else:
#             rolling_avg_inventory = self.inventory  # If not enough data, use current inventory
            
#         # Penalties for excessive changes in cash and inventory
#         cash_change_penalty = -abs(self.cash - rolling_avg_cash) * 0.001
#         inventory_change_penalty = -abs(self.inventory - rolling_avg_inventory) * 0.005
        
#         # Spread capture reward to encourage effective spread capture
#         spread_capture_reward = 0
#         if self.trades:
#             if trade_type == "BUY":
#                 spread_capture_reward = (mid_price - executed_price) * 0.1
#             elif trade_type == "SELL":
#                 spread_capture_reward = (executed_price - mid_price) * 0.1

#         # Small reward for maintaining or increasing PnL over time
#         if len(self.past_pnls) > 1 and pnl > self.past_pnls[-1]:
#             pnl_change_reward = (pnl - self.past_pnls[-1]) * 1 if self.past_pnls else 0
#         else:
#             pnl_change_reward = 0

#         # Adding the market metrics to the reward
#         # These components are meant to encourage the agent to optimize for better market conditions
#         #shortfall_penalty = -abs(implementation_shortfall) * 0.0005
#         #imbalance_reward = -abs(order_flow_imbalance) * 0.0005  # You might want to consider reversing this sign depending on your interpretation
#         # Adding risk management penalties
        
        
#         # Penalize high MAP (large positions) and high drawdown
#         map_penalty = (map_value - 1) * 0.1 if map_value > 1 else 0
#         mdd_penalty = (mdd_value - 0.03) * 0.1 if mdd_value > 0.03 else 0

#         risk_penalty = -(map_penalty + mdd_penalty)

#         # Sharpe ratio bonus: Encourage high risk-adjusted return
#         sharpe_bonus = (sharpe_ratio - 0.5) * 0.05 if sharpe_ratio > 0.5 else 0
        
#         spread_penalty = -abs(half_spread_percentage*100) ** 2 * 0.1

#         is_penalty = -abs(implementation_shortfall) * 0.01
        
#         # Calculate OFI and integrate into reward function
#         ofi = self.order_flow_imbalance()
#         ofi_penalty = -abs(ofi) * 0.001  # Penalty for high OFI (order imbalance)
        
#         # Modify reward calculation to penalize low cash levels
#         min_cash_threshold = 4000  # Minimum cash threshold
#         if self.cash < min_cash_threshold:
#             cash_penalty = -0.001 * (min_cash_threshold - self.cash)  # Penalize low cash levels
#         else:
#             cash_penalty = 0
        
#         # Add inventory holding penalty
#         max_inventory_threshold = 3  # Threshold for maximum desired inventory
#         if self.inventory > max_inventory_threshold:
#             holding_inventory_penalty = -0.01 * (self.inventory - max_inventory_threshold)
#         else:
#             holding_inventory_penalty = 0
        
#         # Modify reward function to focus on long-term profitability
#         if len(self.past_pnls) > 10:
#             long_term_reward = (np.mean(self.past_pnls[-10:]) - np.mean(self.past_pnls[:-10])) * 0.1 
#         else:
#             long_term_reward = 0
        
#         # print(f"pnl: {pnl}, trade_reward: {trade_reward}, inventory_penalty: {inventory_penalty}, cash_change_penalty: {cash_change_penalty}, inventory_change_penalty: {inventory_change_penalty}, execution_quality_reward: {execution_quality_reward}, spread_capture_reward: {spread_capture_reward}, pnl_change_reward: {pnl_change_reward}, spread_penalty: {spread_penalty}, is_penalty: {is_penalty}, sharpe_bonus: {sharpe_bonus}, risk_penalty: {risk_penalty}, ofi_penalty: {ofi_penalty}, cash_penalty: {cash_penalty}, holding_inventory_penalty: {holding_inventory_penalty}, long_term_reward: {long_term_reward}")
#         reward = (
#             pnl 
#             + trade_reward 
#             - inventory_penalty 
#             + cash_change_penalty
#             + inventory_change_penalty
#             + execution_quality_reward 
#             + spread_capture_reward 
#             + pnl_change_reward 
#             + spread_penalty 
#             + is_penalty 
#             + sharpe_bonus 
#             + risk_penalty
#             + ofi_penalty
#             + cash_penalty
#             + holding_inventory_penalty
#             + long_term_reward
#         )
#         if sharpe_ratio < 0.5:
#             reward *= 0.9  # Penalize for poor risk-adjusted returns
#         elif sharpe_ratio > 1:
#             reward *= 1.1  # Reward for good risk-adjusted returns

#         return reward

#     # The _asymmetrical_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
#     def _asymmetrical_reward(self, pnl, action_taken):
#         # Similar to default, but penalize negative pnl more harshly and reward positive pnl slightly more
#         reward = self._default_reward(pnl, action_taken)

#         if pnl < 0:
#             reward *= 1.5  # Increase penalty for negative pnl
#         elif pnl > 0:
#             reward *= 1.1  # Increase reward for positive pnl

#         return reward

#     # The _realized_pnl_reward method focuses purely on the realized PnL from the trades.
#     def _realized_pnl_reward(self, pnl):
#         # Focuses purely on the realized PnL from the trades
#         realized_pnl = sum([trade[1] * trade[2] for trade in self.trades if trade[0] == "SELL"])
#         return realized_pnl * 0.01

#     # The _trade_completion_reward method rewards completing trades and penalizes incomplete trades or open positions.
#     def _trade_completion_reward(self, pnl, action_taken):
#         # Rewards completing trades and penalizes incomplete trades or open positions
#         reward = 0.0

#         if action_taken:
#             reward += 0.02  # Reward for completing a trade

#         if self.inventory == 0:
#             reward += 0.05  # Additional reward for balancing inventory
#         else:
#             reward -= 0.01 * abs(self.inventory)  # Penalize for holding inventory

#         return reward
    
#     def half_spread_percentage(self):
#         # Calculate the percentage of half spread to the current bid price
#         # half_spread_percentage = (self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']) / self.data.iloc[self.current_step]['Bid Price 1']
        
#         # for crypto data
#         half_spread_percentage = (self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']) / self.data.iloc[self.current_step]['BidPrice_0']
#         return half_spread_percentage
    
#     def implementation_shortfall(self):
#         # Implementation Shortfall (IS) calculation
#         if not self.trades:
#             return 0
#         # mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
        
#         # for crypto data
#         mid_price = (self.data.iloc[self.current_step]['BidPrice_0'] + self.data.iloc[self.current_step]['AskPrice_0']) / 2
#         total_is = sum(abs(trade[1] - mid_price) * trade[2] for trade in self.trades)
#         return total_is / len(self.trades) if self.trades else 0

#     # The order_flow_imbalance method calculates the Order Flow Imbalance (OFI) based on the trade flows.
#     def order_flow_imbalance(self):
#         # Order Flow Imbalance (TFI) calculation
#         if not self.trade_flows:
#             return 0
#         total_buy = sum(flow for flow in self.trade_flows if flow > 0)
#         total_sell = sum(flow for flow in self.trade_flows if flow < 0)
#         ofi = (total_buy - total_sell)/10000
#         return ofi
    
#     # The rsi method calculates the Relative Strength Index (RSI) based on the executed sell prices.
#     def rsi(self):
#         # RSI calculation (simple version)
#         prices = [trade[1] for trade in self.trades if trade[0] == "SELL"]
#         if len(prices) < 14:
#             return 50  # Default RSI value
#         gains = [prices[i] - prices[i - 1] for i in range(1, len(prices)) if prices[i] > prices[i - 1]]
#         losses = [-1 * (prices[i] - prices[i - 1]) for i in range(1, len(prices)) if prices[i] < prices[i - 1]]
#         average_gain = np.mean(gains) if gains else 0
#         average_loss = np.mean(losses) if losses else 0
#         rs = average_gain / average_loss if average_loss != 0 else float('inf')
#         return 100 - (100 / (1 + rs))

    
#     def calculate_sharpe_ratio(self):
#         if len(self.past_pnls) <= 1:
#             return 0  # Not enough data to calculate Sharpe ratio
        
#         # Calculate relative returns
#         returns = np.diff(self.past_pnls) / self.past_pnls[:-1]
        
#         if len(returns) == 0:
#             return 0  # Not enough data for returns calculation
        
#         mean_return = np.mean(returns)
#         # check if the return array has more than one elemant
#         if len(returns) < 2:
#             return float('inf') if mean_return > 0 else 0
        
#         std_return = np.std(returns, ddof=1)  # Use sample standard deviation (N-1)
    
        
#         if std_return == 0:
#             return float('inf') if mean_return > 0 else 0  # Return infinite if constant positive returns, otherwise 0
        
#         sharpe_ratio = mean_return / std_return
        
#         return sharpe_ratio

#     def calculate_mean_absolute_position(self):
#         return np.mean(np.abs(self.inventory)) if self.inventory > 0 else 0

#     def calculate_maximum_drawdown(self):
#         peak = np.maximum.accumulate(self.past_pnls)
#         drawdown = (peak - self.past_pnls) / peak
#         return np.max(drawdown) if len(drawdown) > 0 else 0

import numpy as np
from environment.base_env import BaseMarketEnv
from gymnasium import spaces
import time
# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data, reward_type='default'):
        super(ContinuousMarketEnv, self).__init__(data)
        # Adjust action space to include trade size (units) in addition to price adjustments
        self.action_space = spaces.Box(low=np.array([-1, -10]), high=np.array([1, 10]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(89,), dtype=np.float32)
        self.inventory = 0 # Inventory for the agent
        self.cash = 10000 # Cash for the agent
        self.trades = [] # List to store executed trades
        self.reward_type = reward_type
        self.past_pnls = []  # To store past PnL values for risk metrics
        self.trade_flows = []  # To store order flow imbalance data
        self.current_step = 0
        self.action_count = 0
        self.pnl = 0
        self.inventory_penalty = 0
        self.cash_change_penalty = 0
        self.inventory_change_penalty = 0
        self.execution_quality_reward = 0
        self.spread_capture_reward = 0
        self.pnl_change_reward = 0
        self.spread_penalty = 0
        self.is_penalty = 0
        self.sharpe_bonus = 0
        self.risk_penalty = 0
        self.ofi_penalty = 0
        self.cash_penalty = 0
        self.holding_inventory_penalty = 0
        self.long_term_reward = 0
        
        

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self, seed=None):
        if seed is None:
            seed = int(time.time())  # Use current time as the seed
        self.current_step = np.random.randint(0, len(self.data)-501)  # Start from a random step within the first 10 steps
        self.inventory = 0
        self.cash = 10000
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
        
         # Action refinement: scaling trade_size down if conditions met
        if self.calculate_sharpe_ratio() < 1.0:
            trade_size *= 0.5  # Reduce action size for conservative play
            
        # best_bid = self.data.iloc[self.current_step]['Bid Price 1']
        # best_ask = self.data.iloc[self.current_step]['Ask Price 1']
        
        # for crypto data
        best_bid = self.data.iloc[self.current_step]['BidPrice_0']
        best_ask = self.data.iloc[self.current_step]['AskPrice_0']
        mid_price = (best_bid + best_ask) / 2
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
                # print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Sell action (if inventory is sufficient)
            if trade_size < 0 and self.inventory >= abs(trade_size):
                self.inventory += trade_size  # trade_size is negative, so this decreases inventory
                self.cash += abs(trade_size) * executed_ask
                self.trades.append(("SELL", executed_ask, abs(trade_size)))
                action_taken = True
                # print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Increment the action counter when an action is taken
            self.action_count += 1
            
        # Calculate additional metrics for logging and analysis
        sharpe_ratio = self.calculate_sharpe_ratio()
        map_value = self.calculate_mean_absolute_position()
        mdd_value = self.calculate_maximum_drawdown()
        done = self.advance_step()
        
        # Calculate reward based on the action taken
        reward = self.calculate_reward(action_taken)
        state = self.get_current_state()

        # # Calculate the Profit and Loss (PnL)
        # # Realized PnL: Profit from closed trades
        # realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])

        # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
        # unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
        # Store PnL and trade flow data
        # self.past_pnls.append((self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1'])/10000)
        
        #for crypto data
        # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
        unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['Midpoint']

        # Total PnL
        self.pnl = (self.cash + unrealized_pnl) / 10000
        self.past_pnls.append(self.pnl)
        self.trade_flows.append(self.cash) 
        
        # half_spread_percentage = self.half_spread_percentage()
        # implementation_shortfall = self.implementation_shortfall()
        # order_flow_imbalance = self.order_flow_imbalance()
        # rsi = self.rsi()
        
        # Log metrics for debugging and performance analysis
        # print(f"Step: {self.current_step}, Sharpe Ratio: {sharpe_ratio}, MAP: {map_value}, MDD: {mdd_value}")
        
        return state, reward, done, None ,{}
    
    # def get_current_state(self):
    #     # Return the relevant data for the current step
    #     state = np.zeros(35)  # Initialize with zeros
    #     data = self.data.iloc[self.current_step]
        
    #     # Fill in the state with available data
    #     state[:len(data)] = data.values
        
    #     # Add any additional features
    #     state[-5] = self.inventory
    #     state[-4] = self.cash
    #     state[-3] = self.market_depth
    #     state[-2] = self.recent_trade_volume
    #     state[-1] = self.order_flow_imbalance()
        
    #     return state
    # Designed to balance the inventory and execution quality.
    def calculate_reward(self, action_taken):
        # Calculate the percentage of half spread to the current bid price
        half_spread_percentage = self.half_spread_percentage()
        # # Calculate the Profit and Loss (PnL)
        # # Realized PnL: Profit from closed trades
        # realized_pnl = sum([trade[1] * trade[2] if trade[0] == "SELL" else -trade[1] * trade[2] for trade in self.trades])

        # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
        # unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
        
        # for crypto data
        # Calculate the Profit and Loss (PnL)
        # Unrealized PnL: Value of current inventory at the current Bid Price (conservative)
        unrealized_pnl = self.inventory * self.data.iloc[self.current_step]['BidPrice_0']

        # Total PnL
        self.pnl = (self.cash + unrealized_pnl) / 10000  # Normalize by dividing by initial cash
        
        # New metrics calculation
        implementation_shortfall = self.implementation_shortfall()
        map_value = self.calculate_mean_absolute_position()
        mdd_value = self.calculate_maximum_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()

        if self.reward_type == 'default':
            reward = self._default_reward(self.pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio)
        elif self.reward_type == 'asymmetrical':
            reward = self._asymmetrical_reward(self.pnl, action_taken)
        elif self.reward_type == 'realized_pnl':
            reward = self._realized_pnl_reward(self.pnl)
        elif self.reward_type == 'trade_completion':
            reward = self._trade_completion_reward(self.pnl, action_taken)
        elif self.reward_type == 'spread_capture':
            reward = self._spread_capture_reward(self.pnl, action_taken)
        else:
            reward = self._default_reward(self.pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio)  # Fallback to default

        return reward

    # The _default_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
    # This is the main reward function used in the environment.
    def _default_reward(self, pnl, action_taken, half_spread_percentage, implementation_shortfall, map_value, mdd_value, sharpe_ratio):
        window_size = 5
        # Reward for making a trade
        trade_reward = 0.5 if action_taken else 0  # Encourages action

        # Inventory penalty to encourage balanced inventory
        target_inventory = 3  # Adjusted target inventory for simplicity
        dynamic_inventory_penalty = 0.1 if abs(self.inventory - target_inventory) > 2 else 0.01 # Dynamic penalty based on distance from target
        self.inventory_penalty = max(0, abs(self.inventory - target_inventory)) * dynamic_inventory_penalty

        # # Execution quality reward/penalty based on how close the executed price is to the mid-market price
        # mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
        # spread = self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']
        # for crypto data
        mid_price = (self.data.iloc[self.current_step]['BidPrice_0'] + self.data.iloc[self.current_step]['AskPrice_0']) / 2
        spread = self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']
        
        # execution_quality_reward = 0
        # if self.trades:
        #     # Get the last trade details
        #     last_trade = self.trades[-1]
        #     # Unpack the trade details
        #     trade_type, executed_price, trade_size = last_trade

        #     # Penalize execution quality if executed price deviates significantly from mid price
        #     if abs(executed_price - mid_price) > 0.5 * spread:
        #         # Penalize more for larger deviations
        #         execution_quality_reward = -abs(executed_price - mid_price) * 0.05
        #     else:
        #         # Reward for good execution quality
        #         execution_quality_reward = abs(executed_price - mid_price) * 0.05
        
        self.execution_quality_reward = 0
        if self.trades:
            last_trade = self.trades[-1]
            trade_type, executed_price, trade_size = last_trade

            # Calculate slippage: difference between expected execution price and actual execution price
            slippage = abs(executed_price - mid_price)
            
            # Small penalty for slippage based on a fixed percentage of the mid-price
            max_allowed_slippage = 0.001 * mid_price  # Allow up to 1% slippage without penalty
            if slippage > max_allowed_slippage:
                self.execution_quality_reward = -slippage * 0.05  # Penalize only above threshold
            else:
                self.execution_quality_reward = slippage * 0.05  # Small reward for low slippage
                
        # Calculate rolling average cash and inventory
        if len(self.trade_flows) > window_size:
            rolling_avg_cash = np.mean(self.trade_flows[-window_size:])
        else:
            rolling_avg_cash = self.cash  # If not enough data, use current cash

        if len(self.trades) > window_size:
            rolling_avg_inventory = np.mean([trade[2] for trade in self.trades[-window_size:]])
        else:
            rolling_avg_inventory = self.inventory  # If not enough data, use current inventory
            
        # Penalties for excessive changes in cash and inventory
        self.cash_change_penalty = -abs(self.cash - rolling_avg_cash) * 0.0001
        self.inventory_change_penalty = -abs(self.inventory - rolling_avg_inventory) * 0.005
        
        # Spread capture reward to encourage effective spread capture
        self.spread_capture_reward = 0
        if self.trades:
            if trade_type == "BUY":
                self.spread_capture_reward = (mid_price - executed_price) * 0.1
            elif trade_type == "SELL":
                self.spread_capture_reward = (executed_price - mid_price) * 0.1

        # Small reward for maintaining or increasing PnL over time
        if len(self.past_pnls) > 1 and pnl > self.past_pnls[-1]:
            pnl_change_reward = (pnl - self.past_pnls[-1]) * 1 if self.past_pnls else 0
        else:
            self.pnl_change_reward = 0

        # Adding the market metrics to the reward
        # These components are meant to encourage the agent to optimize for better market conditions
        #shortfall_penalty = -abs(implementation_shortfall) * 0.0005
        #imbalance_reward = -abs(order_flow_imbalance) * 0.0005  # You might want to consider reversing this sign depending on your interpretation
        # Adding risk management penalties
        
        
        # Penalize high MAP (large positions) and high drawdown
        map_penalty = (map_value - 1) * 0.1 if map_value > 1 else 0
        mdd_penalty = (mdd_value - 0.03) * 0.1 if mdd_value > 0.03 else 0

        self.risk_penalty = -(map_penalty + mdd_penalty)

        # Sharpe ratio bonus: Encourage high risk-adjusted return
        self.sharpe_bonus = (sharpe_ratio - 0.5) * 0.05 if sharpe_ratio > 0.5 else 0
        
        self.spread_penalty = -abs(half_spread_percentage*100) ** 2 * 0.01

        self.is_penalty = -abs(implementation_shortfall) * 0.01
        
        # Calculate OFI and integrate into reward function
        ofi = self.order_flow_imbalance()
        self.ofi_penalty = -abs(ofi) * 0.001  # Penalty for high OFI (order imbalance)
        
        # Modify reward calculation to penalize low cash levels
        min_cash_threshold = 4000  # Minimum cash threshold
        if self.cash < min_cash_threshold:
            self.cash_penalty = -0.001 * (min_cash_threshold - self.cash)  # Penalize low cash levels
        else:
            self.cash_penalty = 0
        
        # Add inventory holding penalty
        max_inventory_threshold = 3  # Threshold for maximum desired inventory
        if self.inventory > max_inventory_threshold:
            self.holding_inventory_penalty = -0.01 * (self.inventory - max_inventory_threshold)
        else:
            self.holding_inventory_penalty = 0
        
        # Modify reward function to focus on long-term profitability
        if len(self.past_pnls) > 10:
            self.long_term_reward = (np.mean(self.past_pnls[-10:]) - np.mean(self.past_pnls[:-10])) * 0.5
        else:
            self.long_term_reward = 0
        
        #print(f"pnl: {self.pnl}, trade_reward: {trade_reward}, inventory_penalty: {self.inventory_penalty}, cash_change_penalty: {self.cash_change_penalty}, inventory_change_penalty: {self.inventory_change_penalty}, execution_quality_reward: {self.execution_quality_reward}, spread_capture_reward: {self.spread_capture_reward}, pnl_change_reward: {self.pnl_change_reward}, spread_penalty: {self.spread_penalty}, is_penalty: {self.is_penalty}, sharpe_bonus: {self.sharpe_bonus}, risk_penalty: {self.risk_penalty}, ofi_penalty: {self.ofi_penalty}, cash_penalty: {self.cash_penalty}, holding_inventory_penalty: {self.holding_inventory_penalty}, long_term_reward: {self.long_term_reward}")
           
        reward = (
            pnl
            + trade_reward 
            - self.inventory_penalty 
            + self.cash_change_penalty
            + self.inventory_change_penalty
            + self.execution_quality_reward 
            + self.spread_capture_reward 
            + self.pnl_change_reward 
            + self.spread_penalty 
            + self.is_penalty 
            + self.sharpe_bonus 
            + self.risk_penalty
            + self.ofi_penalty
            + self.cash_penalty
            + self.holding_inventory_penalty
            + self.long_term_reward
        )
        if sharpe_ratio < 0.5:
            reward *= 0.9  # Penalize for poor risk-adjusted returns
        elif sharpe_ratio > 1:
            reward *= 1.1  # Reward for good risk-adjusted returns

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
    def _realized_pnl_reward(self, pnl):
        # Focuses purely on the realized PnL from the trades
        realized_pnl = sum([trade[1] * trade[2] for trade in self.trades if trade[0] == "SELL"])
        return realized_pnl * 0.01

    # The _trade_completion_reward method rewards completing trades and penalizes incomplete trades or open positions.
    def _trade_completion_reward(self, pnl, action_taken):
        # Rewards completing trades and penalizes incomplete trades or open positions
        reward = 0.0

        if action_taken:
            reward += 0.02  # Reward for completing a trade

        if self.inventory == 0:
            reward += 0.05  # Additional reward for balancing inventory
        else:
            reward -= 0.01 * abs(self.inventory)  # Penalize for holding inventory

        return reward
    
    def half_spread_percentage(self):
        # Calculate the percentage of half spread to the current bid price
        # half_spread_percentage = (self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']) / self.data.iloc[self.current_step]['Bid Price 1']
        
        # for crypto data
        half_spread_percentage = (self.data.iloc[self.current_step]['AskPrice_0'] - self.data.iloc[self.current_step]['BidPrice_0']) / self.data.iloc[self.current_step]['BidPrice_0']
        return half_spread_percentage
    
    def implementation_shortfall(self):
        # Implementation Shortfall (IS) calculation
        if not self.trades:
            return 0
        # mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
        
        # for crypto data
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
        ofi = (total_buy - total_sell)/10000
        return ofi
    
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

    
    def calculate_sharpe_ratio(self):
        if len(self.past_pnls) <= 1:
            return 0  # Not enough data to calculate Sharpe ratio
        
        # Calculate relative returns
        returns = np.diff(self.past_pnls) / self.past_pnls[:-1]
        
        if len(returns) == 0:
            return 0  # Not enough data for returns calculation
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Use sample standard deviation (N-1)
        
        if std_return == 0:
            return float('inf') if mean_return > 0 else 0  # Return infinite if constant positive returns, otherwise 0
        
        sharpe_ratio = mean_return / std_return
        
        return sharpe_ratio

    def calculate_mean_absolute_position(self):
        return np.mean(np.abs(self.inventory)) if self.inventory > 0 else 0

    def calculate_maximum_drawdown(self):
        peak = np.maximum.accumulate(self.past_pnls)
        drawdown = (peak - self.past_pnls) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    
    