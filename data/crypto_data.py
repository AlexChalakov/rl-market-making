import time
import datetime
from datetime import date, timedelta
import pandas as pd
import ccxt
import matplotlib.pyplot as plt

# Define your start and end dates for training and testing data
startDate_training = date(2024,6,1)
endDate_training = date(2024,7,1)
startDate_test = date(2024,7,1)
endDate_test = date(2024,8,1)

trading_symbol = 'BTC/USDT'
granularity = '1h'

list_dates_training = pd.date_range(startDate_training, endDate_training - timedelta(days=1), freq='d').strftime("%Y-%m-%d").tolist()
list_dates_test = pd.date_range(startDate_test, endDate_test - timedelta(days=1), freq='d').strftime("%Y-%m-%d").tolist()

# Initialize the Binance exchange
exchange = ccxt.binance()

# Prepare empty dataframes for training and test datasets
df_training = pd.DataFrame(columns=['Time', 'BidPrice', 'BidVolume', 'AskPrice', 'AskVolume'])
df_test = pd.DataFrame(columns=['Time', 'BidPrice', 'BidVolume', 'AskPrice', 'AskVolume'])

def fetch_order_book(date, symbol, exchange):
    since = date
    since = time.mktime(datetime.datetime.strptime(since, "%Y-%m-%d").timetuple()) * 1000
    # Fetching order book data
    order_book = exchange.fetch_order_book(symbol)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    bid_price = order_book['bids'][0][0] if order_book['bids'] else None
    bid_volume = order_book['bids'][0][1] if order_book['bids'] else None
    ask_price = order_book['asks'][0][0] if order_book['asks'] else None
    ask_volume = order_book['asks'][0][1] if order_book['asks'] else None
    return {'Time': timestamp, 'BidPrice': bid_price, 'BidVolume': bid_volume, 'AskPrice': ask_price, 'AskVolume': ask_volume}

# Fetch training data
training_data = []
for date in list_dates_training:
    data = fetch_order_book(date, trading_symbol, exchange)
    training_data.append(data)
df_training = pd.concat([df_training, pd.DataFrame(training_data)], ignore_index=True)

# Fetch test data
test_data = []
for date in list_dates_test:
    data = fetch_order_book(date, trading_symbol, exchange)
    test_data.append(data)
df_test = pd.concat([df_test, pd.DataFrame(test_data)], ignore_index=True)

# Save to CSV
df_training.to_csv('order_book_training.csv', index=False)
df_test.to_csv('order_book_test.csv', index=False)