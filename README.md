# Reinforcement Learning for Execution Optimization on Market Making

Reinforcement learning for execution optimization - this project aims to explore and develop a reinforcement learning (RL) model focused on market making and execution optimization in the domain of financial trading. The goal is to create an algorithm that can efficiently balance the dual objectives of inventory risk management and best execution under dynamic market market conditions.

## Project Structure

### Folder Overview
The project is organized into several directories that contain the data, agent implementation, environment, and utility functions for the RL model. Here's a breakdown of each directory and its purpose:

```bash
.
├── agent/                     # Contains the agent implementation using Proximal Policy Optimization (PPO)
│   └── rl_agent.py            # PPO agent class definition, which includes policy updates and actions
│
├── data/                      # Directory containing market data for training and testing
│   ├── crypto/                # Preprocessed crypto limit order book (LOB) data
│   └── lobster/               # LOBSTER data files, including message and order book data
│
├── environment/               # Environment simulating market dynamics for the agent
│   └── env_continuous.py      # Custom environment class (extends OpenAI Gym's environment structure)
│
├── network/                   # Neural network architectures for the policy and value networks
│   └── network.py             # CNN-based policy and value networks used by the PPO agent
│
├── utils/                     # Utility functions for data handling and augmentation
│   └── utils.py               # Includes data loading, augmentation, and preprocessing functions
│
├── results/                   # Directory for storing results such as plots and saved models
│
├── main.py                    # Main script to run the training and evaluation process
├── requirements.txt           # Python dependencies required for running the project
└── README.md                  # Project documentation (this file)
```

## Prerequisites
Python Dependencies
Ensure you have the required Python packages installed. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Running the Code
To run the project, follow these steps:

1. Choose Dataset: Crypto vs. LOBSTER
When you start the main.py script, you will be prompted to choose between the Crypto dataset and the LOBSTER dataset. Input the corresponding number when prompted:
    - 0 for LOBSTER Data 
    - 1 for Crypto Data
2. Execute the Main Script
Run the main.py file to start the training process:
```bash
python main.py
```

3. Training and Evaluation
The script will run through the following steps:

- Load Data: Depending on the selected dataset, the script will load either preprocessed crypto or LOBSTER data.
- Data Splitting: The loaded data is split into training, validation, and test sets.
- Data Augmentation (Optional): The training data can be augmented by adding noise to simulate more volatile environments.
- Environment Setup: The ContinuousMarketEnv class is instantiated with the selected dataset. The environment is responsible for simulating the trading dynamics and interactions with the agent.
- Training: The PPO agent interacts with the environment and learns to balance inventory management and best execution based on the reward structure.
- Evaluation: After training, the agent is evaluated on validation and test datasets, with results being saved to the results/ directory.


## Data Preprocessing
If you want to run the crypto data pipeline again, you can do this with the following command:
```bash
python data/data_pipeline/crypto_data.py
```
