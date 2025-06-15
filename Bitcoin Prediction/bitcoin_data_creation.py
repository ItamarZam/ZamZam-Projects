# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

"""
Bitcoin Data Fetch
"""

import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from binance.client import Client


def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Limit data
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Ensure folders exist
def ensure_folders(base_folder, categories):
    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

# Balance categories by trimming excess files
def balance_categories(base_folder, categories):
    file_counts = {
        category: len(os.listdir(os.path.join(base_folder, category, "input")))
        for category in categories
    }
    min_count = min(file_counts.values())

    for category in categories:
        for subfolder in ["input", "output"]:
            folder_path = os.path.join(base_folder, category, subfolder)
            files = sorted(os.listdir(folder_path))
            for file_to_remove in files[min_count:]:
                os.remove(os.path.join(folder_path, file_to_remove))

# Split data into train, validation, and test sets
def split_data(base_folder, categories, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1 - train_ratio - val_ratio

    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")

        files = sorted(os.listdir(input_folder))
        random.shuffle(files)
        total_files = len(files)

        train_split = int(total_files * train_ratio)
        val_split = int(total_files * (train_ratio + val_ratio))

        train_input_folder = os.path.join(base_folder, "train", category, "input")
        train_output_folder = os.path.join(base_folder, "train", category, "output")
        val_input_folder = os.path.join(base_folder, "validation", category, "input")
        val_output_folder = os.path.join(base_folder, "validation", category, "output")
        test_input_folder = os.path.join(base_folder, "test", category, "input")
        test_output_folder = os.path.join(base_folder, "test", category, "output")

        os.makedirs(train_input_folder, exist_ok=True)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_input_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)
        os.makedirs(test_input_folder, exist_ok=True)
        os.makedirs(test_output_folder, exist_ok=True)

        for i, file in enumerate(files):
            src_input = os.path.join(input_folder, file)
            src_output = os.path.join(output_folder, file.replace("input", "output"))

            if i < train_split:
                dest_input = os.path.join(train_input_folder, file)
                dest_output = os.path.join(train_output_folder, file.replace("input", "output"))
            elif i < val_split:
                dest_input = os.path.join(val_input_folder, file)
                dest_output = os.path.join(val_output_folder, file.replace("input", "output"))
            else:
                dest_input = os.path.join(test_input_folder, file)
                dest_output = os.path.join(test_output_folder, file.replace("input", "output"))

            shutil.move(src_input, dest_input)
            shutil.move(src_output, dest_output)

# Generate candlestick chart pairs and classify them into 'up' and 'down'
def generate_and_categorize(data, base_folder, window_size=24, batch_size=500, resolution=(224, 224)):
    ensure_folders(base_folder, ["up", "down"])

    num_images = len(data) - (2 * window_size) + 1
    total_batches = (num_images + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")

        for i in range(start_idx, end_idx):
            input_subset = data.iloc[i:i + window_size]
            input_subset.set_index('timestamp', inplace=True)

            output_subset = data.iloc[i + window_size:i + 2 * window_size]
            output_subset.set_index('timestamp', inplace=True)

            open_price = output_subset.iloc[0]['open']
            close_price = output_subset.iloc[-1]['close']
            price_change = close_price - open_price

            classification = "up" if price_change > 0 else "down"

            input_folder = os.path.join(base_folder, classification, "input")
            output_folder = os.path.join(base_folder, classification, "output")

            input_file_name = os.path.join(input_folder, f"input_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
            mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
            ax.set_ylabel("Price", fontsize=4)
            ax.tick_params(axis='both', labelsize=5)
            plt.savefig(input_file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

            output_file_name = os.path.join(output_folder, f"output_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
            mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
            ax.set_ylabel("Price", fontsize=4)
            ax.tick_params(axis='both', labelsize=5)
            plt.savefig(output_file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

        print(f"Batch {batch_idx + 1}/{total_batches} completed.")

    print(f"Generated and categorized {num_images} candlestick chart pairs.")


def main_data_creation():
    # API keys (use None if not needed)
    api_key = None
    api_secret = None
    
    # Initialize Binance client
    client = initialize_client(api_key, api_secret)

    # Define parameters for data fetching
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1HOUR
    start_date = "2017-01-03"
    end_date = "2025-03-02"

    print("Fetching data from Binance...")
    data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
    
    # Define base folder for dataset storage
    base_folder = "candlestick_dataset"

    print("Generating and categorizing candlestick images...")
    generate_and_categorize(data, base_folder)

    print("Balancing categories...")
    balance_categories(base_folder, ["up", "down"])

    print("Splitting data into train, validation, and test sets...")
    split_data(base_folder, ["up", "down"])

    print("Process completed successfully!")


def devide_dataset():

    # Original dataset path
    original_dataset = 'candlestick_dataset'
    input_dataset = 'candlestick_dataset_input'
    output_dataset = 'candlestick_dataset_output'

    # Dataset splits
    splits = ['train', 'validation', 'test']
    classes = ['up', 'down']

    # Create new directory structures
    for dataset in [input_dataset, output_dataset]:
        for split in splits:
            for class_name in classes:
                os.makedirs(os.path.join(dataset, split, class_name), exist_ok=True)

    # Move files into the new structure
    for split in splits:
        for class_name in classes:
            input_folder = os.path.join(original_dataset, split, class_name, 'input')
            output_folder = os.path.join(original_dataset, split, class_name, 'output')

            input_target = os.path.join(input_dataset, split, class_name)
            output_target = os.path.join(output_dataset, split, class_name)

            # Move input files
            if os.path.exists(input_folder):
                for file in os.listdir(input_folder):
                    shutil.move(os.path.join(input_folder, file), os.path.join(input_target, file))

            # Move output files
            if os.path.exists(output_folder):
                for file in os.listdir(output_folder):
                    shutil.move(os.path.join(output_folder, file), os.path.join(output_target, file))

    print('Dataset successfully split into input and output datasets.')


def get_downloads_folder():
    return os.path.join(os.path.expanduser("~"), "Downloads")


def generate_single_graph(client, symbol, interval, start_date, resolution=(224, 224)):
    
    # Convert start_date to datetime and compute end_date (24 hours later)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = start_dt + timedelta(days=1)
    end_date = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Fetch data
    data = fetch_hourly_data(client, symbol, interval, start_date, end_date)

    if data.empty:
        print("No data available for the selected timeframe.")
        return None

    # Get the user's Downloads folder dynamically
    downloads_folder = get_downloads_folder()

    # Ensure the Downloads folder exists (it always should, but just in case)
    os.makedirs(downloads_folder, exist_ok=True)

    # File name based on start date
    filename = f"btc_graph_{start_date.replace(':', '-')}.png"
    filepath = os.path.join(downloads_folder, filename)

    # Plot the candlestick chart with matching style
    fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
    mpf.plot(data.set_index('timestamp'), type='candle', ax=ax, style='yahoo', volume=False)

    # Apply the same label styling
    ax.set_ylabel("Price", fontsize=4)
    ax.tick_params(axis='both', labelsize=5)

    # Save the chart
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"Graph saved successfully at {filepath}")

    return filepath




















































"""
6. High resolution for the model I am using 
# Initialize the Binance client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Limit data
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Ensure folders exist
def ensure_folders(base_folder, categories):
    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

# Balance categories by trimming excess files
def balance_categories(base_folder, categories):
    file_counts = {
        category: len(os.listdir(os.path.join(base_folder, category, "input")))
        for category in categories
    }
    min_count = min(file_counts.values())

    for category in categories:
        for subfolder in ["input", "output"]:
            folder_path = os.path.join(base_folder, category, subfolder)
            files = sorted(os.listdir(folder_path))
            for file_to_remove in files[min_count:]:
                os.remove(os.path.join(folder_path, file_to_remove))

# Split data into train, validation, and test sets
def split_data(base_folder, categories, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1 - train_ratio - val_ratio

    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        
        files = sorted(os.listdir(input_folder))
        random.shuffle(files)
        total_files = len(files)

        train_split = int(total_files * train_ratio)
        val_split = int(total_files * (train_ratio + val_ratio))

        train_input_folder = os.path.join(base_folder, "train", category, "input")
        train_output_folder = os.path.join(base_folder, "train", category, "output")
        val_input_folder = os.path.join(base_folder, "validation", category, "input")
        val_output_folder = os.path.join(base_folder, "validation", category, "output")
        test_input_folder = os.path.join(base_folder, "test", category, "input")
        test_output_folder = os.path.join(base_folder, "test", category, "output")

        os.makedirs(train_input_folder, exist_ok=True)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_input_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)
        os.makedirs(test_input_folder, exist_ok=True)
        os.makedirs(test_output_folder, exist_ok=True)

        for i, file in enumerate(files):
            src_input = os.path.join(input_folder, file)
            src_output = os.path.join(output_folder, file.replace("input", "output"))

            if i < train_split:
                dest_input = os.path.join(train_input_folder, file)
                dest_output = os.path.join(train_output_folder, file.replace("input", "output"))
            elif i < val_split:
                dest_input = os.path.join(val_input_folder, file)
                dest_output = os.path.join(val_output_folder, file.replace("input", "output"))
            else:
                dest_input = os.path.join(test_input_folder, file)
                dest_output = os.path.join(test_output_folder, file.replace("input", "output"))

            shutil.move(src_input, dest_input)
            shutil.move(src_output, dest_output)

# Generate candlestick chart pairs and classify them into 'up', 'down', and 'static'
def generate_and_categorize(data, base_folder, window_size=24, batch_size=500, resolution=(400, 300)):
    ensure_folders(base_folder, ["up", "down", "static"])

    num_images = len(data) - (2 * window_size) + 1
    total_batches = (num_images + batch_size - 1) // batch_size  # Calculate total number of batches

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")

        for i in range(start_idx, end_idx):
            input_subset = data.iloc[i:i + window_size]
            input_subset.set_index('timestamp', inplace=True)

            output_subset = data.iloc[i + window_size:i + 2 * window_size]
            output_subset.set_index('timestamp', inplace=True)

            open_price = output_subset.iloc[0]['open']
            close_price = output_subset.iloc[-1]['close']
            price_change = close_price - open_price

            if abs(price_change) < 50:
                classification = "static"
            elif price_change > 0:
                classification = "up"
            else:
                classification = "down"

            input_folder = os.path.join(base_folder, classification, "input")
            output_folder = os.path.join(base_folder, classification, "output")

            # Updated plotting logic to make x-axis labels smaller and less crowded
            input_file_name = os.path.join(input_folder, f"input_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
            mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)

            # Adjust x-axis ticks and labels
            xticks = range(0, len(input_subset.index), max(1, len(input_subset.index) // 5))  # Show 5 evenly spaced ticks
            ax.set_xticks(xticks)
            ax.set_xticklabels(input_subset.index[xticks].strftime('%Y-%m-%d %H:%M'),rotation=30, fontsize=5, ha='right')

            ax.set_xlabel("Time", fontsize = 7)
            ax.set_ylabel("Price (USD)",fontsize = 7)
            plt.savefig(input_file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)

            output_file_name = os.path.join(output_folder, f"output_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
            mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)

            # Adjust x-axis ticks and labels
            xticks = range(0, len(output_subset.index), max(1, len(output_subset.index) // 5))  # Show 5 evenly spaced ticks
            ax.set_xticks(xticks)
            ax.set_xticklabels(
            output_subset.index[xticks].strftime('%Y-%m-%d %H:%M'),rotation=30, fontsize=6, ha='right')

            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            plt.savefig(output_file_name, dpi=100, bbox_inches='tight')
            plt.close(fig)


        print(f"Batch {batch_idx + 1}/{total_batches} completed.")

    print(f"Generated and categorized {num_images} candlestick chart pairs.")


def main_data_creation():
    # Binance API initialization
    api_key = None  # API key (if required)
    api_secret= None  # secret key (if required)
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2023-06-06"
    end_date = "2024-06-05"

    # Folder paths
    main_folder = "C:/Users/itama/Bitcoin Prediction/candlestick_charts"

    try:
        # Fetch data from Binance
        print("Fetching data from Binance...")
        btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
        print(f"Fetched {len(btc_data)} rows of data.")

        # Categorize data and generate charts
        print("Generating and categorizing candlestick charts...")
        generate_and_categorize(btc_data, main_folder, window_size=24, batch_size=500)

        # Balance the data categories
        print("Balancing data categories...")
        balance_categories(main_folder, ["up", "down", "static"])

        # Split data into train, validation, and test sets
        print("Splitting data into train, validation, and test sets...")
        split_data(main_folder, ["up", "down", "static"], train_ratio=0.8, val_ratio=0.1)

        print("Process completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

"""
"""
5. This code is too much for my RAM to handle probably if I take down the image resolution it ,ight be resolved
import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from binance.client import Client

# Initialize the Binance client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Limit data
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Ensure folders exist
def ensure_folders(base_folder, categories):
    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

# Balance categories by trimming excess files
def balance_categories(base_folder, categories):
    file_counts = {
        category: len(os.listdir(os.path.join(base_folder, category, "input")))
        for category in categories
    }
    min_count = min(file_counts.values())

    for category in categories:
        for subfolder in ["input", "output"]:
            folder_path = os.path.join(base_folder, category, subfolder)
            files = sorted(os.listdir(folder_path))
            for file_to_remove in files[min_count:]:
                os.remove(os.path.join(folder_path, file_to_remove))

# Split data into train, validation, and test sets
def split_data(base_folder, categories, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1 - train_ratio - val_ratio

    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        
        files = sorted(os.listdir(input_folder))
        random.shuffle(files)
        total_files = len(files)

        train_split = int(total_files * train_ratio)
        val_split = int(total_files * (train_ratio + val_ratio))

        train_input_folder = os.path.join(base_folder, "train", category, "input")
        train_output_folder = os.path.join(base_folder, "train", category, "output")
        val_input_folder = os.path.join(base_folder, "validation", category, "input")
        val_output_folder = os.path.join(base_folder, "validation", category, "output")
        test_input_folder = os.path.join(base_folder, "test", category, "input")
        test_output_folder = os.path.join(base_folder, "test", category, "output")

        os.makedirs(train_input_folder, exist_ok=True)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_input_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)
        os.makedirs(test_input_folder, exist_ok=True)
        os.makedirs(test_output_folder, exist_ok=True)

        for i, file in enumerate(files):
            src_input = os.path.join(input_folder, file)
            src_output = os.path.join(output_folder, file.replace("input", "output"))

            if i < train_split:
                dest_input = os.path.join(train_input_folder, file)
                dest_output = os.path.join(train_output_folder, file.replace("input", "output"))
            elif i < val_split:
                dest_input = os.path.join(val_input_folder, file)
                dest_output = os.path.join(val_output_folder, file.replace("input", "output"))
            else:
                dest_input = os.path.join(test_input_folder, file)
                dest_output = os.path.join(test_output_folder, file.replace("input", "output"))

            shutil.move(src_input, dest_input)
            shutil.move(src_output, dest_output)

# Generate candlestick chart pairs and classify them into 'up', 'down', and 'static'
def generate_and_categorize(data, base_folder, window_size=24, batch_size=500):
    ensure_folders(base_folder, ["up", "down", "static"])

    num_images = len(data) - (2 * window_size) + 1
    total_batches = (num_images + batch_size - 1) // batch_size  # Calculate total number of batches

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        print(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})...")

        for i in range(start_idx, end_idx):
            input_subset = data.iloc[i:i + window_size]
            input_subset.set_index('timestamp', inplace=True)

            output_subset = data.iloc[i + window_size:i + 2 * window_size]
            output_subset.set_index('timestamp', inplace=True)

            open_price = output_subset.iloc[0]['open']
            close_price = output_subset.iloc[-1]['close']
            price_change = close_price - open_price

            if abs(price_change) < 50:
                classification = "static"
            elif price_change > 0:
                classification = "up"
            else:
                classification = "down"

            input_folder = os.path.join(base_folder, classification, "input")
            output_folder = os.path.join(base_folder, classification, "output")

            input_file_name = os.path.join(input_folder, f"input_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(10, 6))
            mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
            ax.set_xticks(range(len(input_subset.index)))
            ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

            output_file_name = os.path.join(output_folder, f"output_graph_{i + 1:05d}.png")
            fig, ax = plt.subplots(figsize=(10, 6))
            mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
            ax.set_xticks(range(len(output_subset.index)))
            ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Batch {batch_idx + 1}/{total_batches} completed.")

    print(f"Generated and categorized {num_images} candlestick chart pairs.")
"""
"""
4. To much to run simulatiously for RAM
import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from binance.client import Client

# Initialize the Binance client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Limit data
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Ensure folders exist
def ensure_folders(base_folder, categories):
    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

# Balance categories by trimming excess files
def balance_categories(base_folder, categories):
    file_counts = {
        category: len(os.listdir(os.path.join(base_folder, category, "input")))
        for category in categories
    }
    min_count = min(file_counts.values())

    for category in categories:
        for subfolder in ["input", "output"]:
            folder_path = os.path.join(base_folder, category, subfolder)
            files = sorted(os.listdir(folder_path))
            for file_to_remove in files[min_count:]:
                os.remove(os.path.join(folder_path, file_to_remove))

# Split data into train and test sets
def split_data(base_folder, categories, train_ratio=0.8):
    for category in categories:
        input_folder = os.path.join(base_folder, category, "input")
        output_folder = os.path.join(base_folder, category, "output")
        
        files = sorted(os.listdir(input_folder))
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)

        train_input_folder = os.path.join(base_folder, "train", category, "input")
        train_output_folder = os.path.join(base_folder, "train", category, "output")
        test_input_folder = os.path.join(base_folder, "test", category, "input")
        test_output_folder = os.path.join(base_folder, "test", category, "output")
        os.makedirs(train_input_folder, exist_ok=True)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(test_input_folder, exist_ok=True)
        os.makedirs(test_output_folder, exist_ok=True)

        for i, file in enumerate(files):
            src_input = os.path.join(input_folder, file)
            src_output = os.path.join(output_folder, file.replace("input", "output"))

            if i < split_idx:
                dest_input = os.path.join(train_input_folder, file)
                dest_output = os.path.join(train_output_folder, file.replace("input", "output"))
            else:
                dest_input = os.path.join(test_input_folder, file)
                dest_output = os.path.join(test_output_folder, file.replace("input", "output"))

            shutil.move(src_input, dest_input)
            shutil.move(src_output, dest_output)

# Generate candlestick chart pairs and classify them into 'up', 'down', and 'static'
def generate_and_categorize(data, base_folder, window_size=24):
    ensure_folders(base_folder, ["up", "down", "static"])

    num_images = len(data) - (2 * window_size) + 1
    for i in range(num_images):
        input_subset = data.iloc[i:i + window_size]
        input_subset.set_index('timestamp', inplace=True)

        output_subset = data.iloc[i + window_size:i + 2 * window_size]
        output_subset.set_index('timestamp', inplace=True)
        open_price = output_subset.iloc[0]['open']
        close_price = output_subset.iloc[-1]['close']
        price_change = close_price - open_price

        if abs(price_change) < 50:
            classification = "static"
        elif price_change > 0:
            classification = "up"
        else:
            classification = "down"

        input_folder = os.path.join(base_folder, classification, "input")
        output_folder = os.path.join(base_folder, classification, "output")

        input_file_name = os.path.join(input_folder, f"input_graph_{i + 1:05d}.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(input_subset.index)))
        ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_file_name = os.path.join(output_folder, f"output_graph_{i + 1:05d}.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(output_subset.index)))
        ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Generated and categorized {num_images} candlestick chart pairs.")


"""
"""
3.doesnt have the same amount of files
import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import random
from datetime import datetime, timedelta


# Initialize the Binance client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)


# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Limit data
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data


# Generate candlestick chart pairs and classify them into 'up', 'down', and 'static' folders
def generate_candlestick_chart_pairs(data, train_folder, test_folder, window_size=24):
    num_images = len(data) - (2 * window_size) + 1
    indices = list(range(num_images))
    random.shuffle(indices)
    split_index = int(len(indices) * 0.8)  # 80% for training, 20% for testing
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    for i in range(num_images):
        # Determine folder (train or test) based on split
        if i in train_indices:
            base_folder = train_folder
        else:
            base_folder = test_folder

        # Input chart: current day
        input_subset = data.iloc[i:i + window_size]
        input_subset.set_index('timestamp', inplace=True)

        # Output chart: next day
        output_subset = data.iloc[i + window_size:i + 2 * window_size]
        output_subset.set_index('timestamp', inplace=True)
        open_price = output_subset.iloc[0]['open']
        close_price = output_subset.iloc[-1]['close']
        price_change = close_price - open_price

        # Classify into 'up', 'down', or 'static'
        if abs(price_change) < 100:
            classification = "static"
        elif price_change > 0:
            classification = "up"
        else:
            classification = "down"

        # Create folders dynamically
        input_folder = os.path.join(base_folder, classification, "input")
        output_folder = os.path.join(base_folder, classification, "output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        # Save input chart
        input_file_name = os.path.join(input_folder, f"input_graph_{i + 1:05d}.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(input_subset.index)))
        ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save output chart
        output_file_name = os.path.join(output_folder, f"output_graph_{i + 1:05d}.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(output_subset.index)))
        ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Generated {num_images} input-output candlestick chart pairs with classification.")


# Ensure the train/test folder structure
def ensure_classified_folders(base_folder):
    folders = {
        "train": ["up", "down", "static"],
        "test": ["up", "down", "static"]
    }
    for folder_type, classifications in folders.items():
        for classification in classifications:
            os.makedirs(os.path.join(base_folder, folder_type, classification, "input"), exist_ok=True)
            os.makedirs(os.path.join(base_folder, folder_type, classification, "output"), exist_ok=True)

"""
"""
2.with labels
import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import random
from datetime import datetime, timedelta

def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:
            break
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

def generate_candlestick_chart_pairs_with_labels(data, train_folder, test_folder, window_size=24):
    num_images = len(data) - (2 * window_size) + 1
    indices = list(range(num_images))
    random.shuffle(indices)
    split_index = int(len(indices) * 0.8)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    label_data = {'train': [], 'test': []}

    for i in range(num_images):
        if i in train_indices:
            folder = train_folder
            label_set = label_data['train']
        else:
            folder = test_folder
            label_set = label_data['test']

        input_subset = data.iloc[i:i + window_size]
        input_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(input_subset.index)))
        ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        input_file_name = os.path.join(folder, "input", f"input_graph_{i + 1:05d}.png")
        os.makedirs(os.path.dirname(input_file_name), exist_ok=True)
        plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        output_subset = data.iloc[i + window_size:i + 2 * window_size]
        output_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
        ax.set_xticks(range(len(output_subset.index)))
        ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        output_file_name = os.path.join(folder, "output", f"output_graph_{i + 1:05d}.png")
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        first_open_price = output_subset['open'].iloc[0]
        last_close_price = output_subset['close'].iloc[-1]
        price_change = last_close_price - first_open_price
        if abs(price_change) < 100:
            label = "Static"
        elif price_change > 0:
            label = "Up"
        else:
            label = "Down"

        label_set.append({
            "input_file": input_file_name,
            "output_file": output_file_name,
            "label": label
        })

    for label_type, labels in label_data.items():
        label_folder = train_folder if label_type == 'train' else test_folder
        label_csv_path = os.path.join(label_folder, "labels.csv")
        pd.DataFrame(labels).to_csv(label_csv_path, index=False)

    print(f"Generated {num_images} input-output candlestick chart pairs with labels.")

def ensure_train_test_folders(base_folder):
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    os.makedirs(os.path.join(train_folder, "input"), exist_ok=True)
    os.makedirs(os.path.join(train_folder, "output"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "input"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "output"), exist_ok=True)
    return train_folder, test_folder
"""


"""
2.
#this one finally works
import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import shutil
import random
from datetime import datetime, timedelta

# Initialize the Binance client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly Bitcoin data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        # Update start_date to fetch the next batch
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Fetch enough data for 60k images
            break
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Create candlestick chart pairs and save directly in train/test folders
def generate_candlestick_chart_pairs(data, train_folder, test_folder, window_size=24):
    num_images = len(data) - (2 * window_size) + 1
    # Split data into train and test sets
    indices = list(range(num_images))
    random.shuffle(indices)
    split_index = int(len(indices) * 0.8)  # 80% for training, 20% for testing
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    for i in range(num_images):
        # Select train or test folder based on the split
        if i in train_indices:
            folder = train_folder
        else:
            folder = test_folder

        # Input chart: current day
        input_subset = data.iloc[i:i + window_size]
        input_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))  # Larger size for better date visibility
        mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)

        # Customize x-axis
        ax.set_xticks(range(len(input_subset.index)))
        ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")

        # Save input chart
        input_file_name = os.path.join(folder, "input", f"input_graph_{i + 1:05d}.png")
        os.makedirs(os.path.dirname(input_file_name), exist_ok=True)
        plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Output chart: next day
        output_subset = data.iloc[i + window_size:i + 2 * window_size]
        output_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)

        # Customize x-axis
        ax.set_xticks(range(len(output_subset.index)))
        ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")

        # Save output chart
        output_file_name = os.path.join(folder, "output", f"output_graph_{i + 1:05d}.png")
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Generated {num_images} input-output candlestick chart pairs.")

# Ensure that the folder structure is created for train/test data
def ensure_train_test_folders(base_folder):
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")

    # Create folder structure for train and test data
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Create input and output folders within train and test folders
    os.makedirs(os.path.join(train_folder, "input"), exist_ok=True)
    os.makedirs(os.path.join(train_folder, "output"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "input"), exist_ok=True)
    os.makedirs(os.path.join(test_folder, "output"), exist_ok=True)

    return train_folder, test_folder

# Train-test split function to organize files into their respective folders
def train_test_split(train_folder, test_folder):
    # This function is no longer necessary since we're directly saving the files in the correct folders during chart generation
    print("Train-test split completed successfully.")
"""
"""
1.creates the data in output input temorary folders and then transfers them into the train test files(takes more time)
import os
import random
import shutil
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client

# Initialize Binance API client
def initialize_client(api_key=None, api_secret=None):
    return Client(api_key, api_secret)

# Fetch hourly data from Binance
def fetch_hourly_data(client, symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Stop fetching after a sufficient number of rows
            break
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6).iloc[:, :6]  # Ignore extra columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Generate candlestick charts
def generate_candlestick_chart_pairs(data, input_folder, output_folder, window_size=24):
    for i in range(len(data) - 2 * window_size + 1):
        input_data = data.iloc[i:i + window_size].set_index('timestamp')
        output_data = data.iloc[i + window_size:i + 2 * window_size].set_index('timestamp')

        for subset, folder, name in [(input_data, input_folder, f"input_{i+1:05d}.png"),
                                     (output_data, output_folder, f"output_{i+1:05d}.png")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            mpf.plot(subset, type='candle', ax=ax, style="yahoo", volume=False)
            ax.set_xticks(range(len(subset.index)))
            ax.set_xticklabels(subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            plt.savefig(os.path.join(folder, name), dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f"Generated {len(data) - 2 * window_size + 1} candlestick chart pairs.")

# Ensure folders for train/test splits
def create_train_test_folders(base_folder):
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    train_input_folder = os.path.join(train_folder, "input")
    train_output_folder = os.path.join(train_folder, "output")
    test_input_folder = os.path.join(test_folder, "input")
    test_output_folder = os.path.join(test_folder, "output")

    for folder in [train_folder, test_folder, train_input_folder, train_output_folder, test_input_folder, test_output_folder]:
        os.makedirs(folder, exist_ok=True)
    return train_input_folder, train_output_folder, test_input_folder, test_output_folder

# Perform train-test split
def train_test_split(input_folder, output_folder, train_input, train_output, test_input, test_output, test_ratio=0.2):
    
    #Splits data into train and test folders. Ensures folders exist before moving files.
    
    # Get all input graph filenames
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    output_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])

    # Ensure input and output file counts match
    if len(input_files) != len(output_files):
        raise ValueError("Mismatch between input and output file counts.")

    # Shuffle and split indices
    indices = list(range(len(input_files)))
    random.shuffle(indices)
    split_index = int(len(indices) * (1 - test_ratio))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Move files to train and test folders
    print(f"Splitting data into train and test sets: {len(train_indices)} train samples, {len(test_indices)} test samples.")
    for idx in train_indices:
        shutil.move(os.path.join(input_folder, input_files[idx]), os.path.join(train_input, input_files[idx]))
        shutil.move(os.path.join(output_folder, output_files[idx]), os.path.join(train_output, output_files[idx]))

    for idx in test_indices:
        shutil.move(os.path.join(input_folder, input_files[idx]), os.path.join(test_input, input_files[idx]))
        shutil.move(os.path.join(output_folder, output_files[idx]), os.path.join(test_output, output_files[idx]))

    print("Train-test split completed successfully.")
"""


"""
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime, timedelta

# Binance API client
client = Client()  # Add your API key and secret if required

# Function to fetch hourly Bitcoin data in batches
def fetch_hourly_data(symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        # Update start_date to fetch the next batch
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Fetch enough data for 60k images
            break
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Fetch data
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "2024-03-28"
end_date = "2024-03-31"

# Ensure the folder exists
main_folder = "D:\Bitcoin Prediction/candlestick_charts"
input_folder = os.path.join(main_folder, "input")
output_folder = os.path.join(main_folder, "output")
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
print("Folders created!")

print("Fetching data...")
btc_data = fetch_hourly_data(symbol, interval, start_date, end_date)
print(f"Fetched {len(btc_data)} rows of data.")

# Function to generate and save candlestick chart pairs
def generate_candlestick_chart_pairs(data, input_folder, output_folder, window_size=24):
    num_images = len(data) - (2 * window_size) + 1
    
    for i in range(num_images):
        # Input chart: current day
        input_subset = data.iloc[i:i + window_size]
        input_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))  # Larger size for better date visibility
        mpf.plot(input_subset, type='candle', ax=ax, style="yahoo", volume=False)
        
        # Customize x-axis
        ax.set_xticks(range(len(input_subset.index)))
        ax.set_xticklabels(input_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")

        input_file_name = f"{input_folder}/input_graph_{i + 1:05d}.png"
        plt.savefig(input_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Output chart: next day
        output_subset = data.iloc[i + window_size:i + 2 * window_size]
        output_subset.set_index('timestamp', inplace=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(output_subset, type='candle', ax=ax, style="yahoo", volume=False)
        
        # Customize x-axis
        ax.set_xticks(range(len(output_subset.index)))
        ax.set_xticklabels(output_subset.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")

        output_file_name = f"{output_folder}/output_graph_{i + 1:05d}.png"
        plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated {num_images} input-output candlestick chart pairs in {main_folder}")

# Generate candlestick chart pairs
generate_candlestick_chart_pairs(btc_data, input_folder, output_folder, window_size=24)
print(os.getcwd())

#checks the date that the api server is up to date with
server_time = client.get_server_time()
print("Server time:", datetime.fromtimestamp(server_time['serverTime'] / 1000))

"""
"""

Without output graph

from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime, timedelta

# Binance API client
client = Client()  # Add your API key and secret if required

# Function to fetch hourly Bitcoin data in batches
def fetch_hourly_data(symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        # Update start_date to fetch the next batch
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Fetch enough data for 60k images
            break
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Fetch data
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "2021-01-01"
end_date = "2023-12-31"

# Ensure the folder exists
folder_path = "D:/candlestick_charts"
os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
print("folder created!!!")


print("Fetching data...")
btc_data = fetch_hourly_data(symbol, interval, start_date, end_date)
print(f"Fetched {len(btc_data)} rows of data.")

# Function to generate and save candlestick charts
def generate_candlestick_charts(data, folder_path, window_size=24):
    os.makedirs(folder_path, exist_ok=True)
    num_images = len(data) - window_size + 1
    
    for i in range(num_images):
        subset = data.iloc[i:i + window_size]
        subset.set_index('timestamp', inplace=True)
        # Create candlestick chart
        fig, ax = plt.subplots(figsize=(6, 4))  # Consistent size
        mpf.plot(subset, type='candle', ax=ax, style="yahoo", volume=False)
        
        # Save chart
        file_name = f"{folder_path}/chart_{i + 1:05d}.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated {num_images} candlestick charts in {folder_path}")

# Generate candlestick charts
chart_folder = "D:/candlestick_charts"
generate_candlestick_charts(btc_data, chart_folder, window_size=24)
print(os.getcwd())

"""

"""
Day jumps
"""


"""
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime, timedelta

# Binance API client
client = Client()  # Add your API key and secret if required

# Function to fetch hourly Bitcoin data in batches
def fetch_hourly_data(symbol, interval, start_date, end_date):
    all_data = []
    while True:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date, limit=1000)
        if not klines:
            break
        all_data.extend(klines)
        # Update start_date to fetch the next batch
        last_timestamp = klines[-1][0]
        start_date = datetime.utcfromtimestamp(last_timestamp / 1000) + timedelta(milliseconds=1)
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if len(all_data) >= 60000:  # Fetch enough data for 60k images
            break
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = pd.DataFrame(all_data, columns=columns + [None] * 6)  # Extra columns ignored
    data = data.iloc[:, :6]  # Keep relevant columns
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data

# Fetch data
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "2021-01-01"
end_date = "2023-12-31"

# Ensure the folder exists
folder_path = "./candlestick_charts_hourly_by_day"
os.makedirs(folder_path, exist_ok=True)

print("Fetching data...")
btc_data = fetch_hourly_data(symbol, interval, start_date, end_date)
print(f"Fetched {len(btc_data)} rows of data.")

# Set the timestamp as the index
btc_data.set_index('timestamp', inplace=True)

# Function to generate candlestick charts for each day
def generate_hourly_charts_by_day(data, folder_path):
    # Group data by date
    data['date'] = data.index.date
    grouped = data.groupby('date')
    
    for date, group in grouped:
        # Reset index for mplfinance compatibility
        group = group.drop(columns=['date']).copy()
        
        # Skip if data is missing
        if group.empty:
            continue
        
        # Save chart
        file_name = f"{folder_path}/{date}.png"
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
        mpf.plot(group, type='candle', ax=ax, style="yahoo", volume=False, title=f"Bitcoin: {date}")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated charts for {len(grouped)} days in {folder_path}")

# Generate candlestick charts for each day
generate_hourly_charts_by_day(btc_data, folder_path)
print("Charts generation completed.")

"""