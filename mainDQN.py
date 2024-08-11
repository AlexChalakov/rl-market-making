import gym
from stable_baselines3 import DQN  # You can also choose DQN, SAC, A2C, etc.
from stable_baselines3.common.env_checker import check_env
from environment.base_env import BaseMarketEnv

message_file = 'order_book_training.csv'
orderbook_file = 'order_book_test.csv'

# Initialize your environment
env = BaseMarketEnv(
    symbol='USDT_BTC',
    fitting_file= message_file,
    testing_file= orderbook_file,
    max_position=10,
    window_size=100,
    seed=42,
    action_repeats=5,
    training=True,
    format_3d=False,
    reward_type='default',
    transaction_fee=True
)



# Create the RL model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_market_maker")

# Load the trained model
model = DQN.load("ppo_market_maker")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
