# ad_fraud_detection/main.py

import os
import pandas as pd
from dotenv import load_dotenv # Import the load_dotenv function
import data_loader
import feature_engineering
import graph_detector
import ensemble_model
import report_generator

def run_pipeline():
    """
    Executes the full ad fraud detection pipeline.
    """
    # Load environment variables from a .env file
    load_dotenv() 
    
    # Get the Gemini API key from the environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Warning: 'GEMINI_API_KEY' not found in .env file or environment variables.")
        print("The AI-powered report generation will be skipped.")
        print("To enable it, create a .env file with: GEMINI_API_KEY='your_api_key_here'")


    print("Starting Digital Ad Fraud Detection Pipeline...")

    # --- 1. Load Data ---
    data_path = 'data/criteo_sample.csv' 
    print(f"\n[Step 1/5] Loading data from {data_path}...")
    raw_df = data_loader.load_data(data_path, nrows=50000) # Increased nrows for better results
    print(f"Loaded {len(raw_df)} records.")

    # --- 2. Feature Engineering ---
    print("\n[Step 2/5] Engineering session-level and behavioral features...")
    features_df = feature_engineering.create_features(raw_df)
    print("Feature engineering complete.")
    print("Sample of engineered features:")
    print(features_df.head())

    # --- 3. Graph-Based Anomaly Detection ---
    print("\n[Step 3/5] Building graph and detecting anomalies...")
    graph_anomalies = graph_detector.find_graph_anomalies(features_df)
    print(f"Identified {len(graph_anomalies)} suspicious IPs/devices from graph analysis.")
    
    features_df['is_graph_anomaly'] = features_df['ip'].isin(graph_anomalies)


    # --- 4. Ensemble Model Classification ---
    print("\n[Step 4/5] Applying ensemble model for fraud classification...")
    predictions_df = ensemble_model.predict_fraud(features_df)
    print("Classification complete.")
    print("Sample of predictions:")
    print(predictions_df[['ip', 'device_id', 'fraud_probability', 'is_fraud_prediction']].head())

    # --- 5. Generate Fraud Report ---
    print("\n[Step 5/5] Generating automated fraud report...")
    fraudulent_clicks = predictions_df[predictions_df['is_fraud_prediction'] == 1]
    report = report_generator.generate_report(fraudulent_clicks, graph_anomalies, gemini_api_key)
    
    print("\n--- Weekly Fraud Report ---")
    print(report)
    print("--------------------------")

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()
