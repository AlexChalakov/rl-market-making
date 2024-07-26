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

    # Put limit on the number of rows if specified
    if limit is not None and len(data) > limit:
        data = data.sample(n=limit, random_state=42).sort_index()

    return data

def preprocess_data(lob_data):
    # Handle missing data if any
    lob_data = lob_data.fillna(0)
    
    # Normalize prices and volumes
    for col in lob_data.columns:
        if 'Price' in col:
            max_val = lob_data[col].max()
            min_val = lob_data[col].min()
            if max_val != min_val:
                lob_data[col] = (lob_data[col] - min_val) / (max_val - min_val)
            else:
                lob_data[col] = 0  # If max equals min, set normalized value to 0
        if 'Size' in col:
            max_val = lob_data[col].max()
            if max_val != 0:
                lob_data[col] = lob_data[col] / max_val
            else:
                lob_data[col] = 0  # If max is 0, set normalized value to 0
    
    # Calculate additional features
    lob_data['osi'] = lob_data['Bid Size 1'] / (lob_data['Bid Size 1'] + lob_data['Ask Size 1'])
    lob_data['rv'] = lob_data['Size'] / lob_data['Size'].mean()
    
    return lob_data
