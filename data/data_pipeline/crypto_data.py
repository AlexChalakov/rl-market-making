import sys
import os
import time

# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.EMA import ExponentialMovingAverage, apply_ema_all_data

# Configuration Constants
from configurations import MAX_BOOK_ROWS, DATA_PATH, EXCHANGE, EMA_ALPHA, INTERVAL, DURATION

class DataPipeline:
    def __init__(self, max_book_rows=MAX_BOOK_ROWS, ema_alpha=EMA_ALPHA):
        """
        Initialize the DataPipeline with settings for processing the order book data.

        :param max_book_rows: Maximum number of order book levels to consider.
        :param ema_alpha: Alpha value for EMA smoothing.
        """
        self.max_book_rows = max_book_rows
        self.scaler = StandardScaler()
        self.ema = ExponentialMovingAverage(alpha=ema_alpha)

    @staticmethod
    def fetch_order_book(symbol, exchange, max_book_rows=MAX_BOOK_ROWS):
        """
        Fetch the full order book for a symbol.
        
        :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
        :param exchange: CCXT exchange instance (e.g., ccxt.binance()).
        :param max_book_rows: Maximum number of order book levels to capture.
        :return: Dictionary containing full order book data.
        """
        # Fetch order book data from the exchange
        order_book = exchange.fetch_order_book(symbol, limit=max_book_rows)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare data structure to hold the order book
        data = {'Time': timestamp}

        # Capture bid and ask data up to the specified depth
        for i in range(max_book_rows):
            bid_price = order_book['bids'][i][0] if i < len(order_book['bids']) else np.nan
            bid_volume = order_book['bids'][i][1] if i < len(order_book['bids']) else np.nan
            ask_price = order_book['asks'][i][0] if i < len(order_book['asks']) else np.nan
            ask_volume = order_book['asks'][i][1] if i < len(order_book['asks']) else np.nan

            data[f'BidPrice_{i}'] = bid_price
            data[f'BidVolume_{i}'] = bid_volume
            data[f'AskPrice_{i}'] = ask_price
            data[f'AskVolume_{i}'] = ask_volume

        return data

    def collect_lob_data(self, symbol, exchange, interval=INTERVAL, duration=DURATION):
        """
        Collect LOB data at regular intervals and save it.
        
        :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
        :param exchange: CCXT exchange instance (e.g., ccxt.binance()).
        :param interval: Time interval in seconds between each snapshot.
        :param duration: Total duration in seconds to collect data.
        :return: DataFrame containing the LOB snapshots.
        """
        all_data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            lob_data = self.fetch_order_book(symbol, exchange, self.max_book_rows)
            all_data.append(lob_data)
            time.sleep(interval)  # Wait for the next snapshot

        # Convert collected data to DataFrame
        return pd.DataFrame(all_data)

    def preprocess_data(self, data):
        """
        Preprocess the raw order book data, including EMA smoothing and feature engineering.

        :param data: DataFrame containing the raw order book data.
        :return: Preprocessed DataFrame.
        """
        # If the Time column is already in float, there's no need to convert from string
        if data['Time'].dtype == np.float64 or data['Time'].dtype == np.int64:
            pass  # Time is already in the correct format
        else:
            # Convert 'Time' column from string to float if necessary
            data['Time'] = data['Time'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timetuple()))


        # **Normalization Adjustments**
        # Normalize prices by dividing by 10,000 (to bring in line with the LOBSTER data)
        price_columns = [col for col in data.columns if 'Price' in col]
        for col in price_columns:
            data[col] = data[col] / 100

        # Normalize volumes based on the maximum volume in the dataset
        volume_columns = [col for col in data.columns if 'Volume' in col]
        max_volume = data[volume_columns].max().max()
        for col in volume_columns:
            data[col] = data[col] / max_volume

        # Calculate Midpoint from the best bid and ask prices
        data['Midpoint'] = (data['BidPrice_0'] + data['AskPrice_0']) / 2

        # Calculate the bid-ask spread
        data['Spread'] = data['AskPrice_0'] - data['BidPrice_0']

        # Calculate Order Size Imbalance (OSI)
        data['OrderSizeImbalance'] = (data['BidVolume_0'] - data['AskVolume_0']) / (data['BidVolume_0'] + data['AskVolume_0'])

        # Calculate Volume-Weighted Average Price (VWAP)
        data['VWAP'] = ((data['BidPrice_0'] * data['BidVolume_0']) + (data['AskPrice_0'] * data['AskVolume_0'])) / (data['BidVolume_0'] + data['AskVolume_0'])

        # Calculate Relative Volume (RV)
        data['RelativeVolume'] = data['BidVolume_0'] + data['AskVolume_0']

        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Apply EMA smoothing to the 'Midpoint' column
        data = apply_ema_all_data(self.ema, data)

        # Scale only specific features, leaving prices and volumes unscaled
        features_to_scale = ['Spread', 'OrderSizeImbalance', 'VWAP', 'RelativeVolume', 'Midpoint', 'Midpoint_EMA']
        data[features_to_scale] = self.scaler.fit_transform(data[features_to_scale])

        # Ensure all numeric columns are of type float32 for TensorFlow compatibility
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].astype(np.float32)

        return data

    def save_data(self, data, filename):
        """
        Save the processed data to a CSV file.

        :param data: DataFrame containing the data to save.
        :param filename: Path to the output CSV file.
        """
        # Ensure the directory exists
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Save the data
        data.to_csv(os.path.join(DATA_PATH, filename), index=False)

if __name__ == "__main__":
    # Define your symbol
    symbol = 'BTC/USDT'
    
    # Initialize DataPipeline
    pipeline = DataPipeline()

    # Collect LOB data for the specified duration
    lob_data_df = pipeline.collect_lob_data(symbol, EXCHANGE, interval=INTERVAL, duration=DURATION)
    
    # Preprocess the collected data
    processed_data = pipeline.preprocess_data(lob_data_df)
    
    # Save the processed data
    pipeline.save_data(processed_data, 'crypto_lob_data.csv')

    """
    # Load the already collected crypto LOB data
    crypto_data_path = 'data/data_pipeline/crypto_lob_data.csv'
    crypto_data = pd.read_csv(crypto_data_path)
    # Initialize DataPipeline
    pipeline = DataPipeline()
    # Preprocess the collected data with adjustments
    processed_data = pipeline.preprocess_data(crypto_data)
    # Save the processed data
    pipeline.save_data(processed_data, 'processed_crypto_lob_data.csv')
    """