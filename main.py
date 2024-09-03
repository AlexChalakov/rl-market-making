import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from environment.env_continuous import ContinuousMarketEnv
from agent.rl_agent import PPOAgent
from network.network import create_cnn_attention_policy_network, create_cnn_attention_value_network
from utils.utils import load_lobster_data, preprocess_lobster_data, save_preprocessed_data, split_data, augment_data

# Set fixed seed for reproducibility
def set_seed(seed):
    if seed is None:
        seed = int(time.time())  # Use current time as the seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Using seed: {seed}")

# Evaluate the agent on the given data
def evaluate_agent(agent, env, data, title, results_dir):
    env = ContinuousMarketEnv(data)
    state = env.reset()
    done = False
    total_reward = 0
    episode_inventory = []
    episode_cash = []
    episode_rewards = []
    step = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state

        total_reward += reward
        episode_inventory.append(env.inventory)
        episode_cash.append(env.cash)
        episode_rewards.append(reward)
        step += 1

    print(f"{title} - Total Reward: {total_reward}")
    
    # Save the evaluation plots
    plt.figure(figsize=(12, 6))
    plt.plot(episode_cash, label='Cash', color='green')
    plt.xlabel('Step')
    plt.ylabel('Cash')
    plt.title(f'{title} Cash Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_cash_levels.png'))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(episode_inventory, label='Inventory', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Inventory')
    plt.title(f'{title} Inventory Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_inventory_levels.png'))
    plt.show()

    cumulative_reward = np.cumsum(episode_rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_reward, label='Cumulative Reward', color='red')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{title} Cumulative Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_cumulative_reward.png'))
    plt.show()

    return total_reward

def main():
    # Set random seed for reproducibility
    # good seed to test: 1724234196 with lobster data
    set_seed(None) # no fixed seed; use current time

    # *** Default Option: Load Preprocessed Crypto Limit Order Book (LOB) Data ***
    # This option is active by default, loading the preprocessed crypto order book data

    # Uncomment the following lines to use the LOB data
    data_file = os.path.join('data', 'data_pipeline', 'processed_crypto_lob_data.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"The file {data_file} does not exist. Please run the data pipeline first.")

    print("Loading preprocessed crypto order book data...")
    processed_data = pd.read_csv(data_file)

    # *** Alternative Option: Load and Preprocess LOBSTER Data ***
    # Uncomment the following lines to use the LOBSTER data

    # message_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_message_5.csv'
    # orderbook_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv'
    # limit = 20000
    
    # print("Loading and preprocessing LOBSTER data...")
    # lob_data = load_lobster_data(message_file, orderbook_file, limit)
    # processed_data = preprocess_lobster_data(lob_data)

    # Split the data into training, validation, and testing sets
    train_data, val_data, test_data = split_data(processed_data)

    # Optionally apply data augmentation to the training data
    train_data = augment_data(train_data)

    # Save the preprocessed data
    # save_preprocessed_data(processed_data, message_file, orderbook_file)

    # Initialize environment with training data
    print("Initializing Environment, NN and Agent.")
    env = ContinuousMarketEnv(train_data) # ContinuousMarketEnv(processed_data, reward_type='asymmetrical')

    # Define neural network
    input_shape = (train_data.shape[1], 1)  # Adjust based on your data shape
    policy_network = create_cnn_attention_policy_network(input_shape)
    value_network = create_cnn_attention_value_network(input_shape)

    # Initialize agent
    agent = PPOAgent(env, policy_network, value_network)

    # Track metrics
    total_rewards = []
    rewards_per_episode = []
    inventory_per_episode = []
    cash_per_episode = []

    results_dir = "results_crypto"
        # Create a new directory called "results" if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Training loop
    num_episodes = 10  # Adjust as needed
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_inventory = []
        episode_cash = []
        episode_rewards = []
        step = 0
        print(f"Starting episode {episode + 1}/{num_episodes}")
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            episode_inventory.append(env.inventory)
            episode_cash.append(env.cash)
            episode_rewards.append(reward)
            step += 1
            print(f"Step {step}, Reward: {reward}, Total Reward: {total_reward}")

        total_rewards.append(total_reward)
        inventory_per_episode.append(episode_inventory)
        cash_per_episode.append(episode_cash)
        rewards_per_episode.append(episode_rewards)
        print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {total_reward}")

        # Plot Step Rewards Over Time for the current episode
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Step Reward', color='purple')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(f'Step Rewards Over Time - Episode {episode + 1}')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'step_rewards_episode_{episode + 1}.png'))
        # plt.show()

    # Combined plots for episodes 1, 5, and 10
    episodes_to_plot = [0, 4, 9]  # Indexing starts at 0, so 1st, 5th, and 10th episodes are 0, 4, 9

    # Save the trained agent
    print("Saving the trained agent...")
    agent.save(os.path.join(results_dir, 'saved_policy_model.h5'), os.path.join(results_dir, 'saved_value_model.h5'))
    print("Training completed and model saved.")

    # After the training loop, plot the loss
    plt.figure(figsize=(12, 6))
    plt.plot(agent.losses, label='Training Loss', color='blue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'training_loss_over_time.png'))
    plt.show()

    # Plot total rewards per episode with moving average
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', label='Total Reward')
    window = 5  # Define the window size for the moving average
    moving_avg = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(1, len(moving_avg) + 1), moving_avg, color='red', linestyle='--', label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Per Episode')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'total_rewards_per_episode.png'))
    plt.show()

    # Combined Cash Levels Plot
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(cash_per_episode[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Cash')
    plt.title('Cash Levels Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cash_levels_combined.png'))
    plt.show()

    # Combined Inventory Levels Plot
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(inventory_per_episode[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Inventory')
    plt.title('Inventory Levels Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'inventory_levels_combined.png'))
    plt.show()

   # Combined Cumulative Rewards Plot
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        cumulative_reward = np.cumsum(rewards_per_episode[i])
        plt.plot(cumulative_reward, label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cumulative_reward_combined.png'))
    plt.show()

    # Combined Step Rewards Plot
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(range(1, len(rewards_per_episode[i]) + 1), rewards_per_episode[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Step Reward')
    plt.title('Step Rewards Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'step_rewards_combined.png'))
    plt.show()

    # Evaluate the agent on validation data
    val_total_reward = evaluate_agent(agent, env, val_data, "Validation", results_dir)
    print(f"Validation Total Reward: {val_total_reward}")

    # Evaluate the agent on test data
    test_total_reward = evaluate_agent(agent, env, test_data, "Test", results_dir)
    print(f"Test Total Reward: {test_total_reward}")

if __name__ == "__main__":
    main()