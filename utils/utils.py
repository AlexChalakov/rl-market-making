import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_lobster_data(message_file, orderbook_file, limit=None):
    # Load the message and order book files
    messages = pd.read_csv(message_file, header=None)
    orderbook = pd.read_csv(orderbook_file, header=None)
    
    # Assign column names based on the LOBSTER format
    messages.columns = ['Time', 'Type', 'Order ID', 'Size', 'Price', 'Direction']
    orderbook_columns = []
    for level in range(1, 6):  # Assuming 5 levels, adjust if necessary
        orderbook_columns += [f'Ask Price {level}', f'Ask Size {level}', f'Bid Price {level}', f'Bid Size {level}']
    orderbook.columns = orderbook_columns
    
    # Merge messages and order book data
    data = pd.concat([messages, orderbook], axis=1)
    # To visualize the data
    print(data.head())

    if limit is not None:
        data = data.sample(n=limit, random_state=42).sort_index()

    return data

def preprocess_lobster_data(lob_data):
    # Handle missing data if any
    lob_data = lob_data.fillna(0)

    # Normalize prices by dividing by 10,000 to get back to dollar values
    for col in lob_data.columns:
        if 'Price' in col:
            lob_data[col] = lob_data[col] / 10000  # Convert to original dollar value

    # Normalize sizes by the maximum size observed in the dataset
    max_size = lob_data[[col for col in lob_data.columns if 'Size' in col]].max().max()
    for col in lob_data.columns:
        if 'Size' is col:
            lob_data[col] = lob_data[col] / max_size  # Normalize by maximum size

    # Calculate additional features
    lob_data['osi'] = lob_data['Bid Size 1'] / (lob_data['Bid Size 1'] + lob_data['Ask Size 1'])
    lob_data['rv'] = lob_data['Size'] / lob_data['Size'].mean()

    print(lob_data.head())
    return lob_data

def save_preprocessed_data(lob_data, message_file, orderbook_file):
    # Get the directory of the original files
    message_dir = os.path.dirname(message_file)
    orderbook_dir = os.path.dirname(orderbook_file)
    
    # Save the processed data as a CSV in the same directory as the orderbook file
    output_file = os.path.join(orderbook_dir, 'preprocessed_lobster_data.csv')
    lob_data.to_csv(output_file, index=False)
    
    print(f"Preprocessed data saved to {output_file}")

def split_data(data, train_size=0.7, val_size=0.15):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        data (pd.DataFrame): The data to split.
        train_size (float): The proportion of the data to include in the training set.
        val_size (float): The proportion of the data to include in the validation set.
        
    Returns:
        pd.DataFrame: The training data.
        pd.DataFrame: The validation data.
        pd.DataFrame: The test data.
    """
    # Ensure that the sum of train_size and val_size does not exceed 1.0
    if train_size + val_size >= 1.0:
        raise ValueError("train_size and val_size together should be less than 1.0")
    
    # Split the data into training and temporary sets
    train_data, temp_data = train_test_split(data, test_size=1 - train_size, random_state=None, shuffle=True)

    # Calculate the test size as the remaining portion after training and validation
    test_size = 1 - val_size - train_size
    
    # Split the temporary data into validation and test sets
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + val_size), random_state=None, shuffle=True)
    
    return train_data, val_data, test_data

def augment_data(data):
    """
    Applies data augmentation techniques to the data.
    :param data: The input dataset.
    :return: The augmented dataset.
    """
    augmented_data = data.copy()
    
    # Add noise to prices and sizes
    for col in augmented_data.columns:
        if 'Price' in col or 'Size' in col:
            noise = np.random.normal(0, 0.01, augmented_data[col].shape)
            augmented_data[col] += noise
    
    return augmented_data