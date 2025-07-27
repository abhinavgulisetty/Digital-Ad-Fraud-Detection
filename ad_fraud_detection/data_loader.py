# ad_fraud_detection/data_loader.py

import pandas as pd
import numpy as np
from tqdm import tqdm

def _get_criteo_column_names():
    """
    Returns the column names for the Criteo dataset. Kept as a fallback.
    """
    integer_features = [f"I{i}" for i in range(1, 14)]
    categorical_features = [f"C{i}" for i in range(1, 27)]
    return ['label'] + integer_features + categorical_features

def _simulate_fraud_patterns(df, num_bot_ips=5, fraud_percentage=0.1):
    """
    Injects synthetic, realistic fraud patterns into the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with initial simulated columns.
        num_bot_ips (int): The number of IPs to designate as bots.
        fraud_percentage (float): The approximate percentage of total clicks to turn into fraud.

    Returns:
        pd.DataFrame: The DataFrame with injected fraud patterns.
    """
    print(f"\nInjecting synthetic fraud patterns ({fraud_percentage:.0%} of data)...")
    
    num_fraud_records = int(len(df) * fraud_percentage)
    if num_fraud_records == 0:
        print("Not enough data to inject fraud. Skipping.")
        return df

    # 1. Select bot IPs and the records they will take over
    bot_ips = df['ip'].unique()[:num_bot_ips]
    fraud_indices = df.sample(n=num_fraud_records, random_state=42).index

    # 2. Simulate Botnet Behavior
    df.loc[fraud_indices, 'ip'] = np.random.choice(bot_ips, size=num_fraud_records)
    
    # 3. Simulate Device Spoofing
    spoofed_devices = [f"spoofed_device_{i}" for i in range(num_fraud_records)]
    df.loc[fraud_indices, 'device_id'] = np.random.choice(spoofed_devices, size=num_fraud_records)
    df.loc[fraud_indices, 'user_agent'] = np.random.choice(['python-requests/2.28.1', 'curl/7.68.0'], size=num_fraud_records)

    # 4. Simulate Click Stacking / Burst Activity
    base_time = pd.to_datetime('2023-10-27 03:00:00')
    time_deltas_ms = np.random.randint(100, 1000, size=num_fraud_records)
    
    cumulative_deltas = pd.to_timedelta(pd.Series(time_deltas_ms).cumsum(), unit='ms')
    df.loc[fraud_indices, 'timestamp'] = base_time + cumulative_deltas
    
    # Mark these rows with a flag for later analysis
    df['is_injected_fraud'] = 0
    df.loc[fraud_indices, 'is_injected_fraud'] = 1
    
    print(f"Successfully simulated {num_fraud_records} fraudulent clicks from {num_bot_ips} bot IPs.")
    return df


def _simulate_clickstream_columns(df, num_ips=1000, num_devices=1500):
    """
    Adds simulated clickstream columns (IP, device, user agent, timestamp)
    to the DataFrame to mimic a real-world ad-tech scenario.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping simulation.")
        return df

    print("Simulating realistic clickstream columns (IP, device, timestamp)...")
    
    # --- Simulate IP Addresses ---
    ip_pool = [f"10.2.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}" for _ in range(num_ips)]
    ip_probabilities = np.random.power(a=0.1, size=num_ips)
    ip_probabilities /= ip_probabilities.sum()
    df['ip'] = np.random.choice(ip_pool, size=len(df), p=ip_probabilities)

    # --- Simulate Device IDs ---
    device_pool = [f"device_{i}" for i in range(num_devices)]
    device_probabilities = np.random.power(a=0.2, size=num_devices)
    device_probabilities /= device_probabilities.sum()
    df['device_id'] = np.random.choice(device_pool, size=len(df), p=device_probabilities)

    # --- Simulate User Agents ---
    user_agent_pool = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 16_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Linux; Android 12; SM-S906N Build/QP1A.190711.020; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/80.0.3987.119 Mobile Safari/537.36',
    ]
    df['user_agent'] = np.random.choice(user_agent_pool, size=len(df), p=[0.5, 0.4, 0.1])

    # --- Simulate Timestamps ---
    start_time = pd.to_datetime('2023-10-26 00:00:00')
    time_deltas = np.random.randint(1, 30, size=len(df))
    # CORRECTED: Call cumsum() on the numpy array *before* converting to timedelta.
    df['timestamp'] = start_time + pd.to_timedelta(time_deltas.cumsum(), unit='s')
    
    # Standardize column names
    rename_dict = {f"I{i}": f"int_feat_{i}" for i in range(1, 14)}
    rename_dict.update({f"C{i}": f"cat_feat_{i}" for i in range(1, 27)})
    df.rename(columns=rename_dict, inplace=True)

    # Fill NaN values
    for col in [f"int_feat_{i}" for i in range(1, 14)]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in [f"cat_feat_{i}" for i in range(1, 27)]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('missing')

    return df


def load_data(file_path, nrows=100000):
    """
    Loads and preprocesses the Criteo click log data.
    
    Args:
        file_path (str): The path to the gzipped CSV data file.
        nrows (int): The number of rows to load from the file for the sample.
        
    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for feature engineering.
    """
    try:
        # Load the data assuming it's a comma-separated file with a header.
        compression = 'gzip' if file_path.endswith('.gz') else None
        df = pd.read_csv(
            file_path,
            sep=',',
            header=0,
            nrows=nrows,
            compression=compression
        )
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        print("Please ensure the Criteo dataset sample is in the 'data/' directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

    # Add simulated columns to make the dataset suitable for our fraud detection scenario
    df = _simulate_clickstream_columns(df)
    
    # Inject the synthetic fraud patterns
    df = _simulate_fraud_patterns(df, fraud_percentage=0.1)
    
    # Sort by timestamp to make the data chronological
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    return df
