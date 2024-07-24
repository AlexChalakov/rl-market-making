import pandas as pd
import numpy as np

def load_lobster_data(message_file, orderbook_file):
    # Load the message and order book files
    messages = pd.read_csv(message_file, header=None)
    orderbook = pd.read_csv(orderbook_file, header=None)
    
    # Assign column names based on the LOBSTER format
    messages.columns = ['Time', 'Type', 'Order ID', 'Size', 'Price', 'Direction']
    orderbook_columns = []
    for level in range(1, 11):  # Assuming 10 levels, adjust if necessary
        orderbook_columns += [f'Ask Price {level}', f'Ask Size {level}', f'Bid Price {level}', f'Bid Size {level}']
    orderbook.columns = orderbook_columns
    
    # Merge messages and order book data
    data = pd.concat([messages, orderbook], axis=1)
    return data

def preprocess_data(lob_data):
    # Normalize prices and volumes
    for col in lob_data.columns:
        if 'Price' in col:
            lob_data[col] = (lob_data[col] - lob_data[col].min()) / (lob_data[col].max() - lob_data[col].min())
        if 'Size' in col:
            lob_data[col] = lob_data[col] / lob_data[col].max()
    
    # Calculate additional features
    lob_data['osi'] = lob_data['Bid Size 1'] / (lob_data['Bid Size 1'] + lob_data['Ask Size 1'])
    lob_data['rv'] = lob_data['Size'] / lob_data['Size'].mean()
    
    return lob_data
