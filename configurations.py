import datetime
import logging
import ccxt
import pytz as tz

# singleton for logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
LOGGER = logging.getLogger('crypto_rl_log')

# ./data_pipeline/crypto_data.py
MAX_BOOK_ROWS = 20  # Number of levels in the order book to capture
DATA_PATH = "data/data_pipeline"  # Updated path to save data files
TIMEZONE = datetime.timezone.utc  # Set timezone to UTC for consistency
EXCHANGE = ccxt.binance()  # Binance exchange for fetching order book data
EMA_ALPHA = 0.99  # Alpha value for EMA
INTERVAL = 1  # Interval in seconds between LOB snapshots

# UPDATE TO RECORD A FULL 24-HOUR DAY OF DATA
# DURATION = 24 * 60 * 60 # Total duration in seconds for data collection
DURATION = 180  # Total duration in seconds for data collection

# ./indicators/*
INDICATOR_WINDOW = [60 * i for i in [5, 15]]  # Convert minutes to seconds
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = [f'_{i}' for i in [5, 15]]  # Create labels