import os
import pandas as pd
import numpy as np

def load_lobster_data(message_file, orderbook_file, limit = None):
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
    # To visualise the data
    print(data.head())

    # Put limit on the number of rows if specified
    if limit is not None and len(data) > limit:
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
        if 'Size' in col:
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
