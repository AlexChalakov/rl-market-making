import numpy as np
from environment.base_env import BaseMarketEnv
from gym import spaces
from utils.indicators import Indicators
# In a continuous environment, the actions the agent can take are not limited to a finite set but can take any value within a specified range. 

# The ContinuousMarketEnv class extends the BaseMarketEnv class and defines a continuous action space for buying and selling assets.
## This code simulates a market making environment with a focus on inventory management and execution quality.
class ContinuousMarketEnv(BaseMarketEnv):
    def __init__(self, data, reward_type='default'):
        super(ContinuousMarketEnv, self).__init__(data)
        # Adjust action space to include trade size (units) in addition to price adjustments
        self.action_space = spaces.Box(low=np.array([-1, -5]), high=np.array([1, 5]), dtype=np.float32)  # Trade size between -5 and 5 units
        self.inventory = 0 # Inventory for the agent
        self.cash = 10000 # Cash for the agent
        self.trades = [] # List to store executed trades
        self.reward_type = reward_type
        self.past_pnls = []  # To store past PnL values for risk metrics
        self.trade_flows = []  # To store order flow imbalance data
        self.current_step = 0

    # The reset method initializes the environment at the beginning of an episode.
    # It resets the current step, inventory, cash, and trade history.
    def reset(self):
        self.current_step = np.random.randint(0, 10)  # Start from a random step within the first 10 steps
        self.inventory = np.random.randint(0, 5)  # Start with a random inventory level
        self.cash = 9000 + np.random.uniform(-1000, 1000)  # Add a random variation to the cash
        self.trades = []
        self.past_pnls = []
        self.trade_flows = []
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

        # Enhanced action-to-order mapping logic
        bid_adjustment = bid_adjustment * spread
        executed_bid = best_bid + bid_adjustment
        executed_ask = best_ask - bid_adjustment  # Reflecting potential symmetrical adjustments
        
        # Simulate slippage by slightly adjusting the executed price
        slippage = np.random.uniform(0, 0.001)  # Random slippage within a small range
        executed_bid *= (1 + slippage)
        executed_ask *= (1 - slippage)
       
        action_taken = False
        
        # implementing dynamic trade sizing based on recent PnL volatility to be more resilient to adverse market conditions
        recent_pnl_volatility = np.std(self.past_pnls[-10:]) if len(self.past_pnls) > 1 else 0
        trade_size = int(trade_size / (1 + recent_pnl_volatility))  # Reduce size in high volatility

        if trade_size != 0:  # Ensure trade size is not zero
            # Buy action (if cash is sufficient)
            if trade_size > 0 and self.cash >= executed_bid * trade_size:
                # Simulating the effect that large orders can impact prices.  will teach the agent to split orders or be cautious with large trades to minimize impact
                self.data.iloc[self.current_step]['Ask Price 1'] += trade_size * 0.001 
                self.cash -= executed_bid * trade_size
                self.trades.append(("BUY", executed_bid, trade_size))
                action_taken = True
                print(f"BUY: {trade_size} units at {executed_bid}, Inventory: {self.inventory}, Cash: {self.cash}")

            # Sell action (if inventory is sufficient)
            if trade_size < 0 and self.inventory >= abs(trade_size):
                self.data.iloc[self.current_step]['Bid Price 1'] -= abs(trade_size) * 0.001 
                self.inventory += trade_size  # trade_size is negative, so this decreases inventory
                self.cash += abs(trade_size) * executed_ask
                self.trades.append(("SELL", executed_ask, abs(trade_size)))
                action_taken = True
                print(f"SELL: {abs(trade_size)} units at {executed_ask}, Inventory: {self.inventory}, Cash: {self.cash}")

        done = self.advance_step()
        # Calculate reward based on the action taken
        reward = self.calculate_reward(action_taken)

        state = self.get_current_state()
        
        # Store PnL and trade flow data
        self.past_pnls.append(self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1'])
        self.trade_flows.append(self.cash)  # Example, update with actual trade flow imbalance calculation
        
        return state, reward, done, {}

    # Designed to balance the inventory and execution quality.
    def calculate_reward(self, action_taken):
        # coculate the percentage of half spread to the current bid price
        half_spread_percentage = (self.data.iloc[self.current_step]['Ask Price 1'] - self.data.iloc[self.current_step]['Bid Price 1']) / self.data.iloc[self.current_step]['Bid Price 1']
        # Calculate the Profit and Loss (PnL)
        pnl = self.cash + self.inventory * self.data.iloc[self.current_step]['Bid Price 1']
        
        # New metrics calculation
        implementation_shortfall = self.implementation_shortfall()
        order_flow_imbalance = self.order_flow_imbalance()
        rsi = self.rsi()
        mean_average_pricing = self.mean_average_pricing()

        if self.reward_type == 'default':
            reward = self._default_reward(pnl, action_taken, half_spread_percentage, implementation_shortfall, order_flow_imbalance, rsi, mean_average_pricing)

        elif self.reward_type == 'asymmetrical':
            reward = self._asymmetrical_reward(pnl, action_taken)

        elif self.reward_type == 'realized_pnl':
            reward = self._realized_pnl_reward(pnl)

        elif self.reward_type == 'trade_completion':
            reward = self._trade_completion_reward(pnl, action_taken)

        elif self.reward_type == 'spread_capture':
            reward = self._spread_capture_reward(pnl, action_taken)

        else:
            reward = self._default_reward(pnl, action_taken, half_spread_percentage, implementation_shortfall, order_flow_imbalance, rsi, mean_average_pricing)  # Fallback to default

        return reward

    # The _default_reward method calculates the reward based on the Profit and Loss (PnL) and the action taken.
    # This is the main reward function used in the environment.
    def _default_reward(self, pnl, action_taken, half_spread_percentage, implementation_shortfall, order_flow_imbalance, rsi, mean_average_pricing):
        # Reward for making a trade
        trade_reward = 0.1 if action_taken else 0

        # Inventory penalty to encourage balanced inventory (not too high or too low)
        target_inventory = 5  # Define a target inventory level
        inventory_penalty = max(0, abs(self.inventory - target_inventory)) * 0.01  # Reduced penalty for inventory deviation

        # Reward for maintaining some inventory, penalize for zero inventory
        balance_reward = max(self.inventory, 0) * 0.05  # Reduced reward for holding inventory

        # Cash management penalty (discourage negative cash) but less severe
        cash_penalty = max(0, -self.cash) * 0.005  # Slightly reduced penalty for negative cash

        # Initialize mid_price to a default value
        mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2

        # Execution quality reward
        execution_quality_reward = 0
        if self.trades:
            last_trade = self.trades[-1]
            trade_type, executed_price, trade_size = last_trade
            
            # Calculate execution quality reward based on how close the executed price is to the mid-price
            execution_quality_reward = -abs(executed_price - mid_price) * 0.005  # Reduced impact

            if trade_type == "SELL":
                buy_prices = [trade[1] for trade in self.trades if trade[0] == "BUY"]
                if buy_prices:
                    average_buy_price = np.mean(buy_prices)
                    profit_or_loss = executed_price - average_buy_price
                    execution_quality_reward += (profit_or_loss * trade_size * 0.005 if profit_or_loss > 0 else profit_or_loss * trade_size * 0.01)

            # Scale transaction cost based on trade size
            transaction_cost = 0.001 * executed_price * trade_size
            execution_quality_reward -= transaction_cost

        # Risk penalty based on variance of recent PnL values
        recent_pnls = self.past_pnls[-10:]  # Consider the last 10 PnL values
        if len(recent_pnls) > 1:
            pnl_variance = np.var(recent_pnls)
            pnl_mean = np.mean(recent_pnls)
            pnl_std = np.std(recent_pnls)
            risk_penalty = pnl_variance * 0.01  # Adjust the penalty factor as needed
            if pnl_std > 0:
                sharpe_ratio_penalty = pnl_mean / pnl_std
                risk_penalty += sharpe_ratio_penalty * 0.01
        else:
            risk_penalty = 0
        
        # Market volatility and liquidity penalties
        market_volatility = np.std(self.data['Bid Price 1'].iloc[max(0, self.current_step - 10):self.current_step])
        market_liquidity = (self.data.iloc[self.current_step]['Bid Size 1'] + self.data.iloc[self.current_step]['Ask Size 1']) / 2
        holding_penalty = self.inventory * 0.01 * (market_volatility / market_liquidity)
        
        # Time-dependent inventory holding cost
        holding_time_penalty = self.inventory * 0.001 * self.current_step  # Increase with time
        holding_penalty += holding_time_penalty
        
        # Total reward calculation
        reward = pnl - inventory_penalty - cash_penalty - risk_penalty - holding_penalty + execution_quality_reward + balance_reward + trade_reward

        # Additional metrics adjustments
        reward += (half_spread_percentage - 0.01) * 0.01  # Further reduced impact
        reward -= implementation_shortfall * 0.01  # Further reduced penalty
        reward -= abs(order_flow_imbalance) * 0.01  # Further reduced penalty
        reward -= abs(rsi - 50) * 0.005  # Further reduced penalty
        reward -= abs(mean_average_pricing - mid_price) * 0.005  # Further reduced penalty

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

    # The _spread_capture_reward method focuses on capturing the spread between the bid and ask prices.
    def implementation_shortfall(self):
            # Implementation Shortfall (IS) calculation
            if not self.trades:
                return 0
            mid_price = (self.data.iloc[self.current_step]['Bid Price 1'] + self.data.iloc[self.current_step]['Ask Price 1']) / 2
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