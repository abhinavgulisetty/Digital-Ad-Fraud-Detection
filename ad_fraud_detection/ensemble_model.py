# ad_fraud_detection/ensemble_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report # Import the classification_report
import lightgbm as lgb
from tqdm import tqdm

def _create_heuristic_labels(df):
    """
    Creates a proxy 'is_fraud' label based on heuristics for training a model.
    In a real-world scenario, this would be replaced with known fraud labels.
    
    Heuristics used:
    - Clicks from an IP with an extremely high number of clicks.
    - Clicks from an IP connected to a very high number of unique devices.
    - Clicks using a known suspicious user agent.
    
    Args:
        df (pd.DataFrame): The feature-engineered DataFrame.
        
    Returns:
        pd.Series: A boolean series indicating the heuristic fraud label.
    """
    print("Creating heuristic labels for supervised model training...")
    # Define thresholds for what we consider highly suspicious
    ip_click_threshold = df['ip_click_count'].quantile(0.99)
    ip_device_threshold = df['ip_unique_devices'].quantile(0.99)

    # Apply heuristics to create a target variable
    is_fraud_heuristic = (
        (df['ip_click_count'] > ip_click_threshold) |
        (df['ip_unique_devices'] > ip_device_threshold) |
        (df['is_suspicious_ua'] == 1) |
        (df['is_graph_anomaly'] == True)
    )
    
    print(f"Heuristically labeled {is_fraud_heuristic.sum()} out of {len(df)} records as fraud.")
    return is_fraud_heuristic.astype(int)


def predict_fraud(df):
    """
    Uses an ensemble of models to predict the probability of fraud for each click.
    
    The ensemble consists of:
    1. Unsupervised Model (Isolation Forest): To get a general anomaly score.
    2. Supervised Model (LightGBM): Trained on heuristic labels to find known patterns.
    
    Args:
        df (pd.DataFrame): The DataFrame with all engineered features.
        
    Returns:
        pd.DataFrame: The original DataFrame with added 'fraud_probability' 
                      and 'is_fraud_prediction' columns.
    """
    # --- Define Features for Modeling ---
    features_to_use = [
        'int_feat_1', 'int_feat_2', 'int_feat_3', 'int_feat_4', 'int_feat_5', 
        'int_feat_6', 'int_feat_7', 'int_feat_8', 'int_feat_9', 'int_feat_10',
        'int_feat_11', 'int_feat_12', 'int_feat_13',
        'click_hour', 'click_day_of_week', 'is_suspicious_ua', 
        'ip_click_count', 'device_click_count', 'ip_unique_devices', 
        'device_unique_ips', 'clicks_per_hour_per_ip', 
        'ip_device_combo_click_count', 'is_graph_anomaly'
    ]
    X = df[features_to_use].copy()
    
    # --- 1. Unsupervised Anomaly Detection (Isolation Forest) ---
    print("\nRunning Isolation Forest for unsupervised anomaly detection...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(X)
    
    df['unsupervised_anomaly_score'] = -iso_forest.decision_function(X)

    # --- 2. Supervised Classification (LightGBM) ---
    print("\nTraining supervised LightGBM model...")
    y = _create_heuristic_labels(df)
    
    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True)
    lgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5, verbose=False)])
    
    # --- Model Performance Evaluation ---
    print("\n--- Supervised Model Performance ---")
    y_pred_test = lgb_clf.predict(X_test)
    # The 'y_test' here are our heuristic labels, so this report shows how well
    # the model learned to identify the patterns we defined as suspicious.
    print(classification_report(y_test, y_pred_test, target_names=['Legitimate (0)', 'Fraud (1)']))
    print("------------------------------------")


    print("\nApplying supervised model to get fraud probabilities...")
    df['supervised_fraud_prob'] = lgb_clf.predict_proba(X)[:, 1]
    
    # --- 3. Combine Scores for Final Prediction ---
    df['fraud_probability'] = (df['supervised_fraud_prob'] * 0.7) + (df['unsupervised_anomaly_score'] * 0.3)
    
    fraud_threshold = df['fraud_probability'].quantile(0.98) 
    df['is_fraud_prediction'] = (df['fraud_probability'] > fraud_threshold).astype(int)
    
    print(f"\nFinal model flagged {df['is_fraud_prediction'].sum()} records as fraudulent.")
    
    return df
