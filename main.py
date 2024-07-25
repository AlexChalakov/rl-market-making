import pandas as pd
from utils.utils import preprocess_data, load_lobster_data
from environment.env_continuous import ContinuousMarketEnv
from agent.rl_agent import PPOAgent
from network.network import create_cnn_attention_model

def main():
# Load and preprocess data
    message_file = 'data/level 5/AAPL_2012-06-21_34200000_57600000_message_5.csv'
    orderbook_file = 'data/level 5/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv'
    limit = 10000

    lob_data = load_lobster_data(message_file, orderbook_file, limit)
    processed_data = preprocess_data(lob_data)
    print("Loading and preprocessing data...")

    # Initialize environment
    print("Initializing Environment, NN and Agent.")
    env = ContinuousMarketEnv(processed_data)

    # Define neural network
    input_shape = (processed_data.shape[1], 1)  # Adjust based on your data shape
    network = create_cnn_attention_model(input_shape)

    # Initialize agent
    agent = PPOAgent(env, network)

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

        # Optional: Log results after each episode
        print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {total_reward}")

    # Save the trained agent
    print("Saving the trained agent...")
    agent.save('saved_model')
    print("Training completed and model saved.")

    # Evaluate the agent (implement evaluate_agent function if needed)
    # mean_reward, std_reward = evaluate_agent(agent, test_data)
    # print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

if __name__ == "__main__":
    main()
