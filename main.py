import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from environment.env_continuous import ContinuousMarketEnv
from agent.rl_agent import PPOAgent
from network.network import create_cnn_attention_policy_network, create_cnn_attention_value_network
from utils.utils import load_lobster_data, preprocess_lobster_data, save_preprocessed_data 

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # *** Default Option: Load Preprocessed Crypto Limit Order Book (LOB) Data ***
    # This option is active by default, loading the preprocessed crypto order book data

    #data_file = os.path.join('data', 'data_pipeline', 'crypto_lob_data.csv')
    #if not os.path.exists(data_file):
        #raise FileNotFoundError(f"The file {data_file} does not exist. Please run the data pipeline first.")

    #print("Loading preprocessed crypto order book data...")
    #processed_data = pd.read_csv(data_file)

    # *** Alternative Option: Load and Preprocess LOBSTER Data ***
    # Uncomment the following lines to use the LOBSTER data

    message_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_message_5.csv'
    orderbook_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv'
    limit = 10000
    
    print("Loading and preprocessing LOBSTER data...")
    lob_data = load_lobster_data(message_file, orderbook_file, limit)
    processed_data = preprocess_lobster_data(lob_data)

    # Save the preprocessed data
    save_preprocessed_data(processed_data, message_file, orderbook_file)

    # Initialize environment
    print("Initializing Environment, NN and Agent.")
    env = ContinuousMarketEnv(processed_data)
    # Adjust the environment parameters as needed
    # env = ContinuousMarketEnv(processed_data, reward_type='asymmetrical')

    # Define neural network
    input_shape = (processed_data.shape[1], 1)  # Adjust based on your data shape
    policy_network = create_cnn_attention_policy_network(input_shape)
    value_network = create_cnn_attention_value_network(input_shape)

    # Initialize agent
    agent = PPOAgent(env, policy_network, value_network)

    # Track metrics
    total_rewards = []
    inventory_levels = []
    cash_levels = []

    # Training loop
    num_episodes = 10  # Adjust as needed, basic is 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"Starting episode {episode + 1}/{num_episodes}")
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            step += 1
            print(f"Step {step}, Reward: {reward}, Total Reward: {total_reward}")

            # Track inventory and cash after each step
            inventory_levels.append(env.inventory)
            cash_levels.append(env.cash)

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {total_reward}")

    # Create a new directory called "results" if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the trained agent
    print("Saving the trained agent...")
    agent.save(os.path.join(results_dir, 'saved_policy_model.h5'), os.path.join(results_dir, 'saved_value_model.h5'))
    print("Training completed and model saved.")

    # Plot total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'total_rewards_per_episode.png'))
    plt.show()

    # Plot inventory levels over time
    plt.figure(figsize=(12, 6))
    plt.plot(inventory_levels, label='Inventory Level')
    plt.xlabel('Step')
    plt.ylabel('Inventory')
    plt.title('Inventory Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'inventory_levels_over_time.png'))
    plt.show()

    # Plot cash levels over time
    plt.figure(figsize=(12, 6))
    plt.plot(cash_levels, label='Cash Level')
    plt.xlabel('Step')
    plt.ylabel('Cash')
    plt.title('Cash Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cash_levels_over_time.png'))
    plt.show()

if __name__ == "__main__":
    main()