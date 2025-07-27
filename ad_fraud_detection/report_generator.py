# ad_fraud_detection/report_generator.py

import pandas as pd
import os
# Import the Google Generative AI library
import google.generativeai as genai

def _create_prompt(fraud_df, graph_anomalies):
    """
    Creates a detailed prompt for a Generative AI model to summarize the fraud findings.
    
    Args:
        fraud_df (pd.DataFrame): DataFrame containing clicks flagged as fraudulent.
        graph_anomalies (list): List of IPs flagged by the graph analysis.
        
    Returns:
        str: A well-structured prompt for the language model.
    """
    total_fraud_clicks = len(fraud_df)
    
    # Find the top 5 most frequent fraudulent IPs
    top_offenders = fraud_df['ip'].value_counts().nlargest(5).to_dict()
    top_offenders_str = "\n".join([f"- IP: {ip}, Clicks: {count}" for ip, count in top_offenders.items()])
    
    # Describe the key patterns found
    key_patterns = []
    if not fraud_df[fraud_df['is_suspicious_ua'] == 1].empty:
        key_patterns.append("Usage of suspicious user-agents like 'python-requests' and 'curl'.")
    if not fraud_df[fraud_df['is_graph_anomaly'] == True].empty:
        key_patterns.append("Detection of IPs controlling an abnormally high number of devices (graph-based anomaly).")
    
    # Calculate clicks per hour distribution for fraudulent clicks
    hourly_dist = fraud_df['click_hour'].value_counts(normalize=True).sort_index()
    peak_hours = hourly_dist[hourly_dist > hourly_dist.mean()].index.tolist()
    
    key_patterns.append(f"Fraudulent activity was concentrated during these hours (UTC): {peak_hours}.")
    patterns_str = "\n".join([f"- {p}" for p in key_patterns])

    prompt = f"""
    You are an AI assistant for a digital advertising company. Your task is to generate a concise, professional weekly ad fraud report based on the following data.

    **Weekly Fraud Data:**
    - Total Clicks Flagged as Fraudulent: {total_fraud_clicks}
    - Top 5 Fraudulent IP Addresses:
    {top_offenders_str}
    - IPs flagged by graph analysis (controlling multiple devices): {len(graph_anomalies)} IPs
    - Key Patterns Observed:
    {patterns_str}

    **Instructions:**
    1. Start with a high-level summary of the findings.
    2. Detail the top sources of fraudulent activity.
    3. Describe the main characteristics and patterns of the fraud.
    4. Provide clear, actionable recommendations for the security and marketing teams.
    5. Keep the tone professional and data-driven.
    
    Generate the report now.
    """
    return prompt

def _call_gemini_api(prompt, api_key):
    """
    Calls the Gemini API to generate content based on the prompt.
    
    Args:
        prompt (str): The prompt to send to the model.
        api_key (str): The user's Gemini API key.
        
    Returns:
        str: The AI-generated report. Returns an error message on failure.
    """
    print("\n--- Sending Prompt to Gemini API ---")
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content
        response = model.generate_content(prompt)
        
        print("--- Successfully received response from Gemini API ---")
        return response.text
    except Exception as e:
        print(f"--- Error calling Gemini API: {e} ---")
        return f"Error: Could not generate report from Gemini API. Please check your API key and network connection. Details: {e}"


def generate_report(fraud_df, graph_anomalies, api_key):
    """
    Generates a natural language fraud report using the Gemini API.
    
    Args:
        fraud_df (pd.DataFrame): DataFrame containing clicks flagged as fraudulent.
        graph_anomalies (list): List of IPs flagged by the graph analysis.
        api_key (str): The user's Gemini API key.
        
    Returns:
        str: A formatted, human-readable report.
    """
    if fraud_df.empty:
        return "No fraudulent activity was detected this week."
        
    if not api_key:
        return "Error: Gemini API key is missing. Cannot generate report."
        
    # 1. Create the prompt with the latest data
    prompt = _create_prompt(fraud_df, graph_anomalies)
    
    # 2. Call the Gemini API
    report = _call_gemini_api(prompt, api_key)
    
    return report
