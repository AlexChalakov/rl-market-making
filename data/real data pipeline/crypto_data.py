import os
import datetime
import pandas as pd
import numpy as np
import ccxt
from sklearn.preprocessing import StandardScaler
from EMA import ExponentialMovingAverage, apply_ema_all_data  # Assuming EMA.py is in the same directory

# Configuration Constants
MAX_BOOK_ROWS = 15  # Number of levels in the order book to capture
DATA_PATH = "data/real_data_pipeline"  # Updated path to save data files
TIMEZONE = datetime.timezone.utc  # Set timezone to UTC for consistency
EXCHANGE = ccxt.binance()  # Binance exchange for fetching order book data
EMA_ALPHA = 0.99  # Alpha value for EMA

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
    def fetch_order_book(symbol, exchange, date):
        """
        Fetch a snapshot of the order book from the exchange.

        :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
        :param exchange: CCXT exchange instance (e.g., ccxt.binance()).
        :param date: Date string for reference (not used in the function but kept for uniformity).
        :return: Dictionary containing order book data.
        """
        # Fetch order book data from the exchange
        order_book = exchange.fetch_order_book(symbol)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract best bid/ask prices and volumes
        bid_price = order_book['bids'][0][0] if order_book['bids'] else np.nan
        bid_volume = order_book['bids'][0][1] if order_book['bids'] else np.nan
        ask_price = order_book['asks'][0][0] if order_book['asks'] else np.nan
        ask_volume = order_book['asks'][0][1] if order_book['asks'] else np.nan
        midpoint = (bid_price + ask_price) / 2 if bid_price and ask_price else np.nan
        spread = ask_price - bid_price if ask_price and bid_price else np.nan

        # Initialize lists for storing notional values
        bid_notional_list, ask_notional_list = [], []

        # Capture notional values for each level in the order book
        for i in range(MAX_BOOK_ROWS):
            bid_price_i = order_book['bids'][i][0] if i < len(order_book['bids']) else np.nan
            bid_volume_i = order_book['bids'][i][1] if i < len(order_book['bids']) else np.nan
            ask_price_i = order_book['asks'][i][0] if i < len(order_book['asks']) else np.nan
            ask_volume_i = order_book['asks'][i][1] if i < len(order_book['asks']) else np.nan
            bid_notional_list.append(bid_price_i * bid_volume_i if bid_price_i and bid_volume_i else np.nan)
            ask_notional_list.append(ask_price_i * ask_volume_i if ask_price_i and ask_volume_i else np.nan)

        return {
            'Time': timestamp,
            'BidPrice': bid_price,
            'BidVolume': bid_volume,
            'AskPrice': ask_price,
            'AskVolume': ask_volume,
            'Midpoint': midpoint,
            'Spread': spread,
            **{f'BidsNotional_{i}': bid_notional_list[i] for i in range(MAX_BOOK_ROWS)},
            **{f'AsksNotional_{i}': ask_notional_list[i] for i in range(MAX_BOOK_ROWS)},
        }

    def fetch_data_for_dates(self, symbol, dates):
        """
        Fetch order book data for a list of dates.

        :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
        :param dates: List of date strings.
        :return: DataFrame containing the order book data for all dates.
        """
        all_data = []
        for date in dates:
            data = self.fetch_order_book(symbol, EXCHANGE, date)
            all_data.append(data)
        return pd.DataFrame(all_data)

    def preprocess_data(self, data):
        """
        Preprocess the raw order book data, including EMA smoothing.

        :param data: DataFrame containing the raw order book data.
        :return: Preprocessed DataFrame.
        """
        # Ensure all necessary columns are present
        required_columns = ['BidPrice', 'AskPrice', 'BidVolume', 'AskVolume', 'Midpoint', 'Spread']
        for i in range(self.max_book_rows):
            required_columns += [f'BidsNotional_{i}', f'AsksNotional_{i}']

        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Apply EMA smoothing
        data = apply_ema_all_data(self.ema, data)

        # Normalize data
        data[required_columns] = self.scaler.fit_transform(data[required_columns])

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

# Example usage
if __name__ == "__main__":
    # Define your date ranges and symbol
    start_date = datetime.date(2024, 6, 1)
    end_date = datetime.date(2024, 7, 1)
    dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    symbol = 'BTC/USDT'

    # Initialize DataPipeline
    pipeline = DataPipeline()

    # Fetch and preprocess data
    raw_data = pipeline.fetch_data_for_dates(symbol, dates)
    processed_data = pipeline.preprocess_data(raw_data)

    # Save the processed data
    pipeline.save_data(processed_data, 'crypto_order_book_data.csv')