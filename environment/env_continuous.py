import os
import numpy as np
import pandas as pd
from environment.base_env import BaseMarketEnv
from gym import spaces
# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data, data_type='crypto', reward_type='default'):
        super(ContinuousMarketEnv, self).__init__(data)
        # Adjust action space to include trade size (units) in addition to price adjustments
        self.action_space = spaces.Box(low=np.array([-1, -10]), high=np.array([1, 10]), dtype=np.float32)
        self.data_type = data_type
        self.inventory = 0 # Inventory for the agent
        self.cash = 10000 # Cash for the agent
        self.trades = [] # List to store executed trades
        self.reward_type = reward_type
        self.past_pnls = []  # To store past PnL values for risk metrics
        self.trade_flows = []  # To store order flow imbalance data
        self.current_step = 0
        # Same initialization, but we will track more metrics
        self.metrics = {
            'total_pnl': [],  # Track total profit and loss
            'sharpe_ratio': [],  # Track Sharpe ratio
            'total_trades': 0,  # Count the number of trades executed
            'buy_trades': 0,  # Track buy trades
            'sell_trades': 0,  # Track sell trades
            'inventory_levels': [],  # Track inventory levels over time
            'execution_quality': [],  # Deviation from midpoint
            'cumulative_reward': 0,  # Cumulative rewards per episode
            'map': 0,  # Mean Average Pricing
            'ofi': 0,  # Order Flow Imbalance
            'rsi': 0  # Relative Strength Index
        }

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self):
        self.current_step = np.random.randint(0, 10)  # Start from a random step within the first 10 steps
        self.inventory = 0
        self.cash = 9000 + np.random.uniform(-1000, 1000)  # Add a random variation to the cash
        self.trades = []
        self.past_pnls = []
        self.trade_flows = []
        # Reset metrics at the beginning of the episode
        self.metrics = {
            'total_pnl': [],
            'sharpe_ratio': [],
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'inventory_levels': [],
            'execution_quality': [],
            'cumulative_reward': 0,
            'map': 0,
            'ofi': 0,
            'rsi': 0
        }

        return self.data.iloc[self.current_step].values

    # The step method takes an action as input and returns the next state, reward, done flag, and additional information.
    # Simulates order executions. 
    # If the bid adjustment is enough to buy at the best bid price, it buys; 
    # if the ask adjustment is enough to sell at the best ask price, it sells.
    def step(self, action):
        bid_adjustment, trade_size = action

        if self.data_type == 'crypto':
            # Main data columns: 'Bid Price 0', 'Ask Price 0', 'Bid Size 0', 'Ask Size 0'
            # Crypto data uses precomputed spread and midpoint
            best_bid = self.data.iloc[self.current_step]['Bid Price 0']
            best_ask = self.data.iloc[self.current_step]['Ask Price 0']
            bid_size = self.data.iloc[self.current_step]['Bid Size 0']
            ask_size = self.data.iloc[self.current_step]['Ask Size 0']
            midpoint = self.data.iloc[self.current_step]['Midpoint']
            spread = self.data.iloc[self.current_step]['Spread']
            vwap = self.data.iloc[self.current_step]['VWAP']
            osi = self.data.iloc[self.current_step]['OrderSizeImbalance']
            rv = self.data.iloc[self.current_step]['RelativeVolume']
        else:  # 'lobster'
            # LOBSTER data does not have precomputed spread and midpoint
            best_bid = self.data.iloc[self.current_step]['Bid Price 1']
            best_ask = self.data.iloc[self.current_step]['Ask Price 1']
            bid_size = self.data.iloc[self.current_step]['Bid Size 1']
            ask_size = self.data.iloc[self.current_step]['Ask Size 1']
            osi = self.data.iloc[self.current_step]['osi']
            rv = self.data.iloc[self.current_step]['rv']
            spread = best_ask - best_bid
            midpoint = (best_bid + best_ask) / 2

        # Limit trade size based on available liquidity 
        max_trade_size = min(ask_size, bid_size)
        trade_size = np.clip(trade_size * max_trade_size, -ask_size, bid_size)
        #trade_size = np.clip(trade_size * max_trade_size * (self.cash / (self.cash + abs(self.inventory))), -ask_size, bid_size)

        # Enhanced action-to-order mapping logic
        bid_adjustment = bid_adjustment * spread * 1.5
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask - bid_adjustment

        action_taken = False

        if trade_size != 0:  # Ensure trade size is not zero
            # Buy action (if cash is sufficient)
            if trade_size > 0 and self.cash >= executed_bid * trade_size:
                self.inventory += trade_size
                self.cash -= executed_bid * trade_size
                self.trades.append(("BUY", executed_bid, trade_size))
                action_taken = True
                self.metrics['buy_trades'] += 1
                print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Sell action (if inventory is sufficient)
            if trade_size < 0 and self.inventory >= abs(trade_size):
                self.inventory += trade_size  # trade_size is negative, so this decreases inventory
                self.cash += abs(trade_size) * executed_ask
                self.trades.append(("SELL", executed_ask, abs(trade_size)))
                action_taken = True
                self.metrics['sell_trades'] += 1
                print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Increment the action counter when an action is taken
            self.metrics['total_trades'] += 1

        done = self.advance_step()
        # Calculate reward based on the action taken
        reward = self.calculate_reward(action_taken, midpoint, spread)
        
        # Store PnL and trade flow data
        pnl = self.cash + self.inventory * best_bid
        self.past_pnls.append(pnl)
        self.trade_flows.append(self.cash)
        self.metrics['total_pnl'].append(pnl)
        self.metrics['inventory_levels'].append(self.inventory)
        self.metrics['cumulative_reward'] += reward
        sharpe_ratio = self.calculate_sharpe_ratio()
        self.metrics['sharpe_ratio'].append(sharpe_ratio)
        if action_taken:
            last_trade = self.trades[-1]
            trade_type, executed_price, trade_size = last_trade
            execution_quality = abs(executed_price - midpoint)
            self.metrics['execution_quality'].append(execution_quality)
        self.metrics['ofi'] = self.order_flow_imbalance()
        self.metrics['map'] = self.mean_average_pricing()
        self.metrics['rsi'] = self.rsi()

        state = self.get_current_state()
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self, action_taken, midpoint, spread):
        # Calculate PnL based on current cash and inventory value
        if self.data_type == 'crypto':
            pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 0']
        else:
            pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
                
        # Smoothing PnL changes over the last few steps (to avoid sharp fluctuations)
        #if len(self.past_pnls) > 10:
            #smoothed_pnl = np.mean(self.past_pnls[-10:])
        #else:
            #smoothed_pnl = pnl

        if self.reward_type == 'default':
            reward = self._default_reward(pnl, action_taken, midpoint, spread)
        elif self.reward_type == 'asymmetrical':
            reward = self._asymmetrical_reward(pnl, action_taken)
        elif self.reward_type == 'realized_pnl':
            reward = self._realized_pnl_reward(pnl)
        elif self.reward_type == 'trade_completion':
            reward = self._trade_completion_reward(pnl, action_taken)
        elif self.reward_type == 'spread_capture':
            reward = self._spread_capture_reward(pnl, action_taken)
        else:
            reward = self._default_reward(pnl, action_taken, spread)

        return reward

    # The _default_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
    # This is the main reward function used in the environment.
    def _default_reward(self, pnl, action_taken, midpoint, spread):
        # Reward for making a trade
        trade_reward = 1.0 if action_taken else 0  # Encourages action, slightly reduced

        # Inventory penalty to encourage balanced inventory
        target_inventory = 2  # Adjusted target inventory for simplicity
        dynamic_inventory_penalty = 0.1 if abs(self.inventory - target_inventory) > 2 else 0.05  # Further reduced
        inventory_penalty = max(0, abs(self.inventory - target_inventory)) * dynamic_inventory_penalty

        # Execution quality reward/penalty based on how close the executed price is to the midpoint
        #midpoint = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
        execution_quality_reward = 0
        if self.trades:
            # Get the last trade details
            last_trade = self.trades[-1]
            # Unpack the trade details
            trade_type, executed_price, trade_size = last_trade

            # Reduce the impact of execution quality
            if abs(executed_price - midpoint) > 0.5 * spread:
                execution_quality_reward = -abs(executed_price - midpoint) * 0.1  # Reduced further
            else:
                execution_quality_reward = -abs(executed_price - midpoint) * 0.05  # Reduced further

        # Spread capture reward to encourage effective spread capture
        spread_capture_reward = 0
        if self.trades:
            if trade_type == "BUY":
                spread_capture_reward = (midpoint - executed_price) * 0.05  # Reduced
            elif trade_type == "SELL":
                spread_capture_reward = (executed_price - midpoint) * 0.05  # Reduced

        pnl_change_reward = (pnl - self.past_pnls[-1]) * 0.01 if len(self.past_pnls) > 1 else 0

        # Smoothed PnL change reward (reduces volatility)
        #if len(self.past_pnls) > 1:
            #pnl_change_reward = (pnl - self.past_pnls[-1]) * 0.005  # Smaller impact to smooth out fluctuations
        #else:
            #pnl_change_reward = 0

        # Sharpe Ratio Penalty (Risk-Adjusted Reward)
        sharpe_ratio = self.calculate_sharpe_ratio()
        if sharpe_ratio < 0.5:
            reward_adjustment = 0.9
        elif sharpe_ratio > 1:
            reward_adjustment = 1.1
        else:
            reward_adjustment = 1.0

        # Total reward is the sum of the components
        reward = (
            pnl
            + trade_reward
            - inventory_penalty
            + execution_quality_reward
            + spread_capture_reward
            + pnl_change_reward
        )

        """
        # Print the reward components if an action was taken
        if action_taken:
            print(f"Action Taken: {'BUY' if trade_type == 'BUY' else 'SELL'}")
            print(f"PnL: {pnl}")
            print(f"Trade Reward: {trade_reward}")
            print(f"Inventory Penalty: {inventory_penalty}")
            print(f"Execution Quality Reward: {execution_quality_reward}")
            print(f"Spread Capture Reward: {spread_capture_reward}")
            print(f"PnL Change Reward: {pnl_change_reward}")
            print(f"Total Step Reward: {reward}")
        """

        # Adjust reward based on Sharpe Ratio
        reward *= reward_adjustment  
        
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

    # The calculate_sharpe_ratio method calculates the Sharpe Ratio as a risk-adjusted performance metric.
    def calculate_sharpe_ratio(self):
        # Calculate Sharpe Ratio as a risk-adjusted performance metric
        returns = np.diff(self.past_pnls)  # Daily returns
        if len(returns) < 2:
            return 0
        avg_return = np.mean(returns)
        return_volatility = np.std(returns)
        sharpe_ratio = avg_return / return_volatility if return_volatility != 0 else 0
        return sharpe_ratio
    
    # The order_flow_imbalance method calculates the Order Flow Imbalance (OFI) based on the trade flows.
    def order_flow_imbalance(self):
        if not self.trade_flows:
            return 0
        total_buy = sum(flow for flow in self.trade_flows if flow > 0)
        total_sell = sum(flow for flow in self.trade_flows if flow < 0)
        return total_buy - total_sell
    
    # The rsi method calculates the Relative Strength Index (RSI) based on the executed sell prices.
    def rsi(self):
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
        if not self.trades:
            return 0
        total_price = sum(trade[1] * trade[2] for trade in self.trades)
        total_quantity = sum(trade[2] for trade in self.trades)
        return total_price / total_quantity if total_quantity else 0
    
    # Get best bid field
    def get_best_bid(self):
        if self.data_type == 'crypto':
            return self.data.iloc[self.current_step]['Bid Price 0']
        else:
            return self.data.iloc[self.current_step]['Bid Price 1']
        
    def save_metrics(self, episode_num, file_path="metrics_log.csv"):
        # Convert metrics to a DataFrame
        metrics_df = pd.DataFrame({
            'Episode': [episode_num],
            'Total PnL': [sum(self.metrics['total_pnl'])],
            'Average Sharpe Ratio': [np.mean(self.metrics['sharpe_ratio']) if self.metrics['sharpe_ratio'] else 0],
            'Total Trades': [self.metrics['total_trades']],
            'Buy Trades': [self.metrics['buy_trades']],
            'Sell Trades': [self.metrics['sell_trades']],
            'Average Inventory Level': [np.mean(self.metrics['inventory_levels']) if self.metrics['inventory_levels'] else 0],
            'Execution Quality (Avg)': [np.mean(self.metrics['execution_quality']) if self.metrics['execution_quality'] else 0],
            'Cumulative Reward': [self.metrics['cumulative_reward']],
            'MAP': [self.metrics['map']],
            'OFI': [self.metrics['ofi']],
            'RSI': [self.metrics['rsi']]
        })
        
        # Append the DataFrame to the CSV file, or create the file if it doesn't exist
        if not os.path.isfile(file_path):
            metrics_df.to_csv(file_path, mode='w', header=True, index=False)
        else:
            metrics_df.to_csv(file_path, mode='a', header=False, index=False)