# ad_fraud_detection/feature_engineering.py

import pandas as pd
from tqdm import tqdm

def create_features(df):
    """
    Engineers a rich set of features from the raw click data to detect fraud.

    Args:
        df (pd.DataFrame): The input DataFrame with raw and simulated columns.

    Returns:
        pd.DataFrame: A DataFrame with newly engineered features ready for modeling.
    """
    print("Starting feature engineering...")

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- 1. Time-Based Features ---
    # Extracting time components can reveal patterns (e.g., bots active at odd hours).
    df['click_hour'] = df['timestamp'].dt.hour
    df['click_day_of_week'] = df['timestamp'].dt.dayofweek

    # --- 2. User-Agent Features ---
    # Flag known bot-like user agents.
    suspicious_ua_keywords = ['python-requests', 'curl', 'bot', 'headless']
    ua_pattern = '|'.join(suspicious_ua_keywords)
    df['is_suspicious_ua'] = df['user_agent'].str.contains(ua_pattern, case=False, na=False).astype(int)

    # --- 3. Click Frequency and Aggregation Features ---
    # These features are critical for identifying automation. We calculate counts
    # over different groups to spot anomalies. Using `transform` is efficient
    # as it returns a series with the same index as the original DataFrame.

    # tqdm.pandas() allows us to see progress bars for pandas operations.
    tqdm.pandas(desc="Calculating IP click counts")
    
    # a) Clicks per IP: High counts can indicate a bot or a large NAT.
    df['ip_click_count'] = df.groupby('ip')['ip'].progress_transform(lambda x: x.count())

    # b) Clicks per Device: High counts from a single device are suspicious.
    tqdm.pandas(desc="Calculating Device click counts")
    df['device_click_count'] = df.groupby('device_id')['device_id'].progress_transform(lambda x: x.count())
    
    # c) Unique devices per IP: A single IP with many devices can be a sign of a botnet.
    tqdm.pandas(desc="Calculating unique devices per IP")
    df['ip_unique_devices'] = df.groupby('ip')['device_id'].progress_transform(lambda x: x.nunique())

    # d) Unique IPs per device: A device rapidly changing IPs is a red flag (proxy switching).
    tqdm.pandas(desc="Calculating unique IPs per device")
    df['device_unique_ips'] = df.groupby('device_id')['ip'].progress_transform(lambda x: x.nunique())
    
    # e) Clicks from an IP in a given hour: Bursts of activity are suspicious.
    tqdm.pandas(desc="Calculating clicks per hour per IP")
    df['clicks_per_hour_per_ip'] = df.groupby(['ip', 'click_hour'])['ip'].progress_transform(lambda x: x.count())

    # --- 4. Combination Features ---
    # Combine IP and device to create a more unique identifier.
    df['ip_device_combo'] = df['ip'].astype(str) + '_' + df['device_id'].astype(str)
    
    tqdm.pandas(desc="Calculating combo click counts")
    df['ip_device_combo_click_count'] = df.groupby('ip_device_combo')['ip_device_combo'].progress_transform(lambda x: x.count())

    print("Feature engineering completed.")
    
    # For modeling, we'll need to select the relevant feature columns.
    # The original categorical and high-cardinality columns will be dropped later.
    return df
