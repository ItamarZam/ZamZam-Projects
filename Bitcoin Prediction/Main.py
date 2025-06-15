# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:16:49 2025

@author: itama
"""
from bitcoin_data_creation import (
    main_data_creation,
    devide_dataset,
    generate_single_graph
)
from model import(
    run_model,
    )

def main():
    run_model()    
if __name__=="__main__":    
    main()

"""
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_and_categorize,
    balance_categories,
    split_data
)
from model import(
    run_model,
    )

def main():
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

if __name__ == "__main__":
    main()

"""
"""
5.
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_and_categorize,
    balance_categories,
    split_data
)

def main():
    # Binance API initialization
    api_key = None  # Add your API key if required
    api_secret = None  # Add your secret key if required
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2024-05-27"
    end_date = "2024-05-30"

    # Folder paths
    main_folder = "C:/Users/itama/Bitcoin Prediction/candlestick_charts"

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
    

if __name__ == "__main__":
    main()

"""
"""
4.
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_and_categorize,
    balance_categories,
    split_data
)

def main():
    # Binance API initialization
    api_key = None  # Add your API key if required
    api_secret = None  # Add your secret key if required
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2018-05-27"
    end_date = "2024-05-30"

    # Folder paths
    main_folder = "D:/Bitcoin Prediction/candlestick_charts"

    # Fetch data from Binance
    print("Fetching data from Binance...")
    btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
    print(f"Fetched {len(btc_data)} rows of data.")

    # Categorize data and generate charts
    print("Generating and categorizing candlestick charts...")
    generate_and_categorize(btc_data, main_folder)

    # Balance the data categories
    print("Balancing data categories...")
    balance_categories(main_folder, ["up", "down", "static"])

    # Split data into train and test sets
    print("Splitting data into train and test sets...")
    split_data(main_folder, ["up", "down", "static"])

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
"""
"""
3.
import os
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_candlestick_chart_pairs,
    ensure_classified_folders
)


def main():
    # Binance API initialization
    api_key = None  # Add your API key if required
    api_secret = None  # Add your secret key if required
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2018-05-27"
    end_date = "2018-05-30"

    # Folder paths
    main_folder = "D:/Bitcoin Prediction/candlestick_charts"

    # Ensure train and test folder structure exists
    ensure_classified_folders(main_folder)

    # Fetch data from Binance
    print("Fetching data from Binance...")
    btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
    print(f"Fetched {len(btc_data)} rows of data.")

    # Generate candlestick chart pairs and classify them
    train_folder = f"{main_folder}/train"
    test_folder = f"{main_folder}/test"

    print("Generating and classifying candlestick chart pairs...")
    generate_candlestick_chart_pairs(btc_data, train_folder, test_folder, window_size=24)

    print("Process completed successfully.")


if __name__ == "__main__":
    main()


"""
"""
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_candlestick_chart_pairs_with_labels,
    ensure_train_test_folders
)

def main():
    api_key = None  # Add your Binance API key if required
    api_secret = None  # Add your Binance secret key if required
    client = initialize_client(api_key, api_secret)

    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2024-05-28"
    end_date = "2024-05-31"

    main_folder = "D:/Bitcoin Prediction/candlestick_charts"
    train_folder, test_folder = ensure_train_test_folders(main_folder)

    print("Fetching data from Binance...")
    btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
    print(f"Fetched {len(btc_data)} rows of data.")

    print("Generating candlestick chart pairs with labels...")
    generate_candlestick_chart_pairs_with_labels(btc_data, train_folder, test_folder, window_size=24)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()

""""""""

2.without the labels 
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_candlestick_chart_pairs,
    ensure_train_test_folders
)
def main():
    # Binance API initialization
    api_key = None  # Add your API key if required
    api_secret = None  # Add your secret key if required
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2018-05-28"
    end_date = "2024-05-31"

    # Folder paths
    main_folder = "D:/Bitcoin Prediction/candlestick_charts"

    # Ensure train and test folder structure exists
    train_folder, test_folder = ensure_train_test_folders(main_folder)

    # Fetch data from Binance
    print("Fetching data from Binance...")
    btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
    print(f"Fetched {len(btc_data)} rows of data.")

    # Generate candlestick chart pairs and save directly to train/test folders
    print("Generating candlestick chart pairs...")
    generate_candlestick_chart_pairs(btc_data, train_folder, test_folder, window_size=24)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()

"""""""

1.
import os
from bitcoin_data_creation import (
    initialize_client,
    fetch_hourly_data,
    generate_candlestick_chart_pairs,
    train_test_split,
)

def create_train_test_folders(base_folder):
    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    train_input_folder = os.path.join(train_folder, "input")
    train_output_folder = os.path.join(train_folder, "output")
    test_input_folder = os.path.join(test_folder, "input")
    test_output_folder = os.path.join(test_folder, "output")

    for folder in [train_folder, test_folder, train_input_folder, train_output_folder, test_input_folder, test_output_folder]:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

    return train_input_folder, train_output_folder, test_input_folder, test_output_folder

def main():
    # Binance API initialization
    api_key = None  # Add your API key if required
    api_secret = None  # Add your secret key if required
    client = initialize_client(api_key, api_secret)

    # Parameters for fetching data
    symbol = "BTCUSDT"
    interval = "1h"
    start_date = "2024-05-28"
    end_date = "2024-05-31"

    # Folder paths
    main_folder = "D:/Bitcoin Prediction/candlestick_charts"
    train_input, train_output, test_input, test_output = create_train_test_folders(main_folder)

    # Temporary input/output folders
    temp_input_folder = os.path.join(main_folder, "temp_input")
    temp_output_folder = os.path.join(main_folder, "temp_output")
    os.makedirs(temp_input_folder, exist_ok=True)
    os.makedirs(temp_output_folder, exist_ok=True)

    # Check if data already exists to avoid fetching again
    if not os.listdir(temp_input_folder) or not os.listdir(temp_output_folder):
        print("Fetching data from Binance...")
        btc_data = fetch_hourly_data(client, symbol, interval, start_date, end_date)
        print(f"Fetched {len(btc_data)} rows of data.")

        # Generate candlestick chart pairs
        print("Generating candlestick chart pairs...")
        generate_candlestick_chart_pairs(btc_data, temp_input_folder, temp_output_folder, window_size=24)
    else:
        print("Candlestick charts already exist in temporary folders. Skipping data fetching and generation.")

    # Perform train-test split
    print("Performing train-test split...")
    train_test_split(
        temp_input_folder, 
        temp_output_folder, 
        train_input, 
        train_output, 
        test_input, 
        test_output, 
        test_ratio=0.2
    )

    # Clean up temporary folders
    os.rmdir(temp_input_folder)
    os.rmdir(temp_output_folder)

    # Check server time
    server_time = get_server_time(client)
    print("Server time:", server_time)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
"""