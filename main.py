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
from utils.utils import augment_data, load_lobster_data, preprocess_lobster_data, save_preprocessed_data, split_data

# Set fixed seed for reproducibility
def set_seed(seed):
    if seed is None:
        seed = int(time.time())  # Use current time as the seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Using seed: {seed}")

# Evaluate the agent on the given data
def evaluate_agent(agent, env, data, title, results_dir, plot_sharpe=False):
    env.data = data
    state = env.reset()
    done = False
    total_reward = 0
    episode_inventory = []
    episode_cash = []
    episode_rewards = []
    sharpe_ratios = []
    step = 0
    buy_steps = []
    sell_steps = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state

        total_reward += reward
        episode_inventory.append(env.inventory)
        episode_cash.append(env.cash)
        episode_rewards.append(reward)

        # Collect steps for "BUY" and "SELL" to plot
        if len(env.trades) > 0:
            last_trade = env.trades[-1]
            trade_type, executed_price, trade_size = last_trade
            if trade_type == "BUY":
                buy_steps.append(step)
            elif trade_type == "SELL":
                sell_steps.append(step)

        # Sharpe ratio calculation if required
        if plot_sharpe:
            sharpe_ratio = env.calculate_sharpe_ratio()
            sharpe_ratios.append(sharpe_ratio)

        step += 1

    print(f"{title} - Total Reward: {total_reward}")

    # Step rewards plot with green dots for SELL and red dots for BUY
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Step Reward', color='purple')
    plt.scatter(buy_steps, [episode_rewards[i] for i in buy_steps], color='red', label='BUY', marker='o', s=20)
    plt.scatter(sell_steps, [episode_rewards[i] for i in sell_steps], color='green', label='SELL', marker='o', s=20)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'{title} Step Rewards over Time with Actions')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_step_rewards_with_actions.png'))
    #plt.show()

    # Overlay Two Lines: One for Buys and One for Sells
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Step Reward', color='purple')
    # Overlay buy and sell points with different markers and lines
    plt.plot(buy_steps, [episode_rewards[i] for i in buy_steps], 'r-', label='BUY')
    plt.plot(sell_steps, [episode_rewards[i] for i in sell_steps], 'g-', label='SELL')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'{title} Step Rewards Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_step_rewards_with_actions_overlay2lines.png'))
    #plt.show()

    # Sharpe ratio over time (for Episode 1, 5, 10 or when plot_sharpe=True)
    if plot_sharpe:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(sharpe_ratios) + 1), sharpe_ratios, label='Sharpe Ratio', color='orange')
        plt.xlabel('Step')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'{title} Sharpe Ratio Over Time')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_sharpe_ratio.png'))
        #plt.show()

    # PnL over time
    plt.figure(figsize=(12, 6))
    plt.plot(episode_cash, label='Cash', color='green')
    plt.xlabel('Step')
    plt.ylabel('Cash')
    plt.title(f'{title} Cash Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_cash_levels.png'))
    #plt.show()

    # Inventory levels over time
    plt.figure(figsize=(12, 6))
    plt.plot(episode_inventory, label='Inventory', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Inventory')
    plt.title(f'{title} Inventory Levels Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_inventory_levels.png'))
    #plt.show()

    # Cumulative reward plot
    cumulative_reward = np.cumsum(episode_rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_reward, label='Cumulative Reward', color='red')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{title} Cumulative Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'{title.lower().replace(" ", "_")}_cumulative_reward.png'))
    #plt.show()

    return total_reward, sharpe_ratios

# Evaluate on validation and test data
def evaluate_on_validation_test(agent, env, val_data, test_data, results_dir):
    val_total_reward = evaluate_agent(agent, env, val_data, "Validation", results_dir, plot_sharpe=False)
    print(f"Validation Total Reward: {val_total_reward}")

    test_total_reward = evaluate_agent(agent, env, test_data, "Test", results_dir, plot_sharpe=False) 
    print(f"Test Total Reward: {test_total_reward}")

# Main function to train the agent
def main():
    # Set random seed for reproducibility
    set_seed(None)  # no fixed seed; use current time

    # Prompt user for dataset selection
    print("Select dataset to load:")
    print("0: LOBSTER Data")
    print("1: Crypto Data")
    data_choice = input("Enter 0 for LOBSTER or 1 for Crypto: ")

    processed_data = None
    data_type = "crypto"

    # Load the selected dataset
    if data_choice == "0":
        # *** Alternative Option: Load and Preprocess LOBSTER Data ***
        # Load and preprocess LOBSTER data
        message_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_message_5.csv'
        orderbook_file = 'data/experimentation/level 5/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv'
        limit = 20000
        
        print("Loading and preprocessing LOBSTER data...")
        lob_data = load_lobster_data(message_file, orderbook_file, limit)
        processed_data = preprocess_lobster_data(lob_data)

        data_type = "lobster"
        results_dir = "results_lobster"
        
    elif data_choice == "1":
        # *** Default Option: Load Preprocessed Crypto Limit Order Book (LOB) Data ***
        # Load preprocessed Crypto Limit Order Book (LOB) data
        data_file = os.path.join('data', 'data_pipeline', 'processed_crypto_lob_data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"The file {data_file} does not exist. Please run the data pipeline first.")
        
        print("Loading preprocessed crypto order book data...")
        processed_data = pd.read_csv(data_file)
        
        data_type = "crypto"
        results_dir = "results_crypto"
        
    else:
        print("Invalid selection. Please enter 0 for LOBSTER or 1 for Crypto.")
        return  # Exit the program if invalid input is given

    # Create the results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Split the data into training, validation, and testing sets
    train_data, val_data, test_data = split_data(processed_data)

    # Optionally apply data augmentation to the training data
    train_data = augment_data(train_data)

    # Save the preprocessed data
    # save_preprocessed_data(processed_data, message_file, orderbook_file)

    # Initialize environment with training data
    print("Initializing Environment, NN and Agent.")
    env = ContinuousMarketEnv(train_data, data_type=data_type)
    # Adjust the environment parameters as needed
    #env = ContinuousMarketEnv(processed_data, reward_type='asymmetrical')

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
    best_bid_per_episode = []
    sharpe_ratios_dict = {}

    # Training loop
    num_episodes = 10  # Adjust as needed
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_inventory = []
        episode_cash = []
        episode_rewards = []
        episode_best_bids = []
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
            best_bid = env.get_best_bid()
            episode_best_bids.append(best_bid)
            step += 1
            print(f"Step {step}, Reward: {reward}, Total Reward: {total_reward}")

        total_rewards.append(total_reward)
        inventory_per_episode.append(episode_inventory)
        cash_per_episode.append(episode_cash)
        rewards_per_episode.append(episode_rewards)
        best_bid_per_episode.append(episode_best_bids)
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

        # Plot Step Rewards with BUY/SELL markers for episodes 1, 5, and 10
        if episode + 1 in [1, 5, 10]:
            _, sharpe_ratios = evaluate_agent(agent, env, train_data, f"Episode {episode + 1}", results_dir, plot_sharpe=True)
            sharpe_ratios_dict[f'Episode {episode + 1}'] = sharpe_ratios

    # Save the trained agent
    print("Saving the trained agent...")
    agent.save(os.path.join(results_dir, 'saved_policy_model.h5'), os.path.join(results_dir, 'saved_value_model.h5'))
    print("Training completed and model saved.")

    # Combined Sharpe Ratio plot for Episodes 1, 5, 10 (you can refine this after Episode 1 works)
    plt.figure(figsize=(12, 6))
    for ep_num in sharpe_ratios_dict:
        plt.plot(sharpe_ratios_dict[ep_num], label=f'Episode {ep_num}')
    plt.xlabel('Step')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'sharpe_ratio_combined.png'))
    #plt.show()

  # Plot PnL over time for all 10 episodes
    plt.figure(figsize=(12, 6))
    for ep_num in range(num_episodes):
        pnl = [cash_per_episode[ep_num][step] + inventory_per_episode[ep_num][step] * current_bid_price for step, current_bid_price in enumerate(best_bid_per_episode[ep_num])]
        plt.plot(pnl, label=f'Episode {ep_num + 1}')
        
    plt.xlabel('Step')
    plt.ylabel('PnL')
    plt.title('Profit and Loss (PnL) Over Time for All Episodes')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'pnl_over_time_all_episodes.png'))
    #plt.show()

    # After the training loop, plot the loss
    plt.figure(figsize=(12, 6))
    plt.plot(agent.losses, label='Training Loss', color='blue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'training_loss_over_time.png'))
    #plt.show()

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
    #plt.show()

    # Evaluate the agent on validation and test data
    evaluate_on_validation_test(agent, env, val_data, test_data, results_dir)

if __name__ == "__main__":
    main()
