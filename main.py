import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from environment.env_continuous import ContinuousMarketEnv
from gymnasium.wrappers import TimeLimit
from agent.rl_agent import PPOAgent, TWAPBaseline
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
def run_twap_baseline(env, twap_model):
    """
    Run the TWAP baseline model in the provided environment.

    Parameters:
    - env: The trading environment instance.
    - twap_model: The TWAP baseline model instance.
    """
    state = env.reset()
    twap_model.reset()

    done = False
    while not done:
        trade_size = twap_model.act()  # TWAP decides the trade size for this step
        action = np.array([0, trade_size])  # Assuming 0 price adjustment and TWAP decides trade size
        state, reward, done, _ = env.step(action)

    print("TWAP Baseline Run Complete")
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
    env = TimeLimit(env, max_episode_steps=500)
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
    sharpe_ratios = []
    maps = []
    mdds = []
    ofis = []
    
    # Track all actions taken by the agent
    all_actions = []
    
     # Initialize TWAPBaseline
    total_quantity = 100  # Example total quantity to trade
    execution_time = 500  # Example execution time (same as max_episode_steps)
    twap_agent = TWAPBaseline(env, total_quantity, execution_time)
    
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
        episode_actions = []
        episode_inventory = []
        episode_cash = []
        episode_rewards = []
        episode_sharpe_ratios = []
        episode_maps = []
        episode_mdds = []
        # cumulative metrics for evaulating the effctiveness of the reward function
        pnl = []
        inventory_penalty = []
        cash_change_penalty = []
        inventory_change_penalty = []
        execution_quality_reward = []
        spread_capture_reward= []
        pnl_change_reward = []
        spread_penalty = []
        is_penalty = []
        sharpe_bonus = []
        risk_penalty = []
        ofi_penalty= []
        cash_penalty = []
        holding_inventory_penalty = []
        long_term_reward = []
        step = 0
        print(f"Starting episode {episode + 1}/{num_episodes}")
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            episode_inventory.append(env.inventory)
            episode_cash.append(env.cash)
            episode_rewards.append(reward)
            episode_actions.append(action)
            
            # Collect metrics from environment
            sharpe_ratio = env.calculate_sharpe_ratio()
            map_value = env.calculate_mean_absolute_position()
            mdd_value = env.calculate_maximum_drawdown()
            
            episode_sharpe_ratios.append(sharpe_ratio)
            episode_maps.append(map_value)
            episode_mdds.append(mdd_value)
            
            pnl.append(env.pnl)
            inventory_penalty.append(env.inventory_penalty)
            cash_change_penalty.append(env.cash_change_penalty)
            inventory_change_penalty.append(env.inventory_change_penalty)
            execution_quality_reward.append(env.execution_quality_reward)
            spread_capture_reward.append(env.spread_capture_reward)
            pnl_change_reward.append(env.pnl_change_reward)
            spread_penalty.append(env.spread_penalty)
            is_penalty.append(env.is_penalty)
            sharpe_bonus.append(env.sharpe_bonus)
            risk_penalty.append(env.risk_penalty)
            ofi_penalty.append(env.ofi_penalty)
            cash_penalty.append(env.cash_penalty)
            holding_inventory_penalty.append(env.holding_inventory_penalty)
            long_term_reward.append(env.long_term_reward)
            
            step += 1
            #print(f"Step {step}, Reward: {reward}, Total Reward: {total_reward}")
        all_actions.extend(episode_actions)
        total_rewards.append(total_reward)
        inventory_per_episode.append(episode_inventory)
        cash_per_episode.append(episode_cash)
        rewards_per_episode.append(episode_rewards)
        sharpe_ratios.append(episode_sharpe_ratios)
        maps.append(episode_maps)
        mdds.append(episode_mdds)
        print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {total_reward}")

        # Plot Step Rewards Over Time for the every 5 episodes
        if (episode + 1) % 5 == 0:
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
    # Combined sharpe ratio plots
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(sharpe_ratios[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Over Time - Episodes 1, 5, 10')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'sharpe_ratio_combined.png'))
    
    # Combined MAP plots
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(maps[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('MAP')
    plt.title('Mean Absolute Position (MAP) Over Time - Episodes 1, 20, 100, 300, 560')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'map_combined.png'))
    
    # Combined MDD plots
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(mdds[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('MDD')
    plt.title('Maximum Drawdown (MDD) Over Time - Episodes 1, 20, 100, 300, 560')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'mdd_combined.png'))
    
    # Plot Order Flow Imbalance (OFI) Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(ofis, label='Order Flow Imbalance (OFI)')
    plt.xlabel('Step')
    plt.ylabel('OFI')
    plt.title('Order Flow Imbalance (OFI) Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'ofi_over_time.png'))
    plt.show()
    
    # plot every metrics in the reward function
    # Plot a graoh of inventory_penalty in every episode
    plt.figure(figsize=(12, 6))
    plt.plot(inventory_penalty, label='Inventory Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Inventory Penalty')
    plt.title('Inventory Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'inventory_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(cash_change_penalty, label='Cash Change Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Cash Change Penalty')
    plt.title('Cash Change Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cash_change_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(inventory_change_penalty, label='Inventory Change Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Inventory Change Penalty')
    plt.title('Inventory Change Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'inventory_change_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(execution_quality_reward, label='Execution Quality Reward', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Execution Quality Reward')
    plt.title('Execution Quality Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'execution_quality_reward_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(spread_capture_reward, label='Spread Capture Reward', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Spread Capture Reward')
    plt.title('Spread Capture Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'spread_capture_reward_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_change_reward, label='PnL Change Reward', color='blue')
    plt.xlabel('Step')
    plt.ylabel('PnL Change Reward')
    plt.title('PnL Change Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'pnl_change_reward_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(spread_penalty, label='Spread Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Spread Penalty')
    plt.title('Spread Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'spread_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(is_penalty, label='Is Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Is Penalty')
    plt.title('Is Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'is_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(sharpe_bonus, label='Sharpe Bonus', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Sharpe Bonus')
    plt.title('Sharpe Bonus Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'sharpe_bonus_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(risk_penalty, label='Risk Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Risk Penalty')
    plt.title('Risk Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'risk_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(ofi_penalty, label='OFI Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('OFI Penalty')
    plt.title('OFI Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'ofi_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(cash_penalty, label='Cash Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Cash Penalty')
    plt.title('Cash Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cash_penalty_over_time.png'))

    plt.figure(figsize=(12, 6))
    plt.plot(holding_inventory_penalty, label='Holding Inventory Penalty', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Holding Inventory Penalty')
    plt.title('Holding Inventory Penalty Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'holding_inventory_penalty_over_time.png'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(long_term_reward, label='Long Term Reward', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Long Term Reward')
    plt.title('Long Term Reward Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'long_term_reward_over_time.png'))
    
    # After the training loop, plot the loss
    plt.figure(figsize=(12, 6))
    plt.plot(agent.losses, label='Training Loss', color='blue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'training_loss_over_time.png'))
    plt.show()

    # Plot Profit and Loss (P&L) Over Time
    plt.figure(figsize=(10, 6))
    for episode, rewards in enumerate(rewards_per_episode):
        plt.plot(np.cumsum(rewards), label=f'Episode {episode + 1}')
    plt.title('Profit and Loss (P&L) Over Time')
    plt.xlabel('Step')
    plt.ylabel('P&L')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'profit_loss_over_time.png'))
    
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
    
    # Scatter Plot of Actions (using actual actions)
    plt.figure(figsize=(12, 6))
    actions = np.array(all_actions)
    plt.scatter(range(len(actions)), actions[:, 1], label='Actions', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Action')
    plt.title('Actions Taken Over Time')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'actions_over_time.png'))
    
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
    plt.title('Cumulative Reward Over Time - Episodes 1, 10, 20, 30')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'cumulative_reward_combined.png'))
    plt.show()

    # Combined Step Rewards Plot
    plt.figure(figsize=(12, 6))
    for i in episodes_to_plot:
        plt.plot(range(1, len(rewards_per_episode[i]) + 1), rewards_per_episode[i], label=f'Episode {i + 1}')
    plt.xlabel('Step')
    plt.ylabel('Step Reward')
    plt.title('Step Rewards Over Time - Episodes 1, 10, 20, 30')
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