
# Digital Ad Fraud Detection Project

This project provides a complete, end-to-end pipeline for detecting digital advertising fraud from clickstream data. It leverages session-based feature engineering, graph-based anomaly detection, an ensemble machine learning model, and Generative AI for automated, human-readable reporting.

---

## Problem Overview

Digital ad fraud, including fake clicks and impressions from bots, drains billions from advertisers' budgets, skews marketing data, and reduces campaign ROI. This project tackles this issue by identifying sophisticated, automated fraud patterns that go beyond simple rule-based systems.

---

## Key Features

- **Rich Feature Engineering:** Creates behavioral features from raw click data, analyzing timing, frequency, user environment, and relationships between IPs and devices.
- **Graph-Based Anomaly Detection:** Models the data as a network of IPs and devices to uncover coordinated botnet activity that is invisible to row-by-row analysis.
- **Advanced Ensemble Model:** Combines an unsupervised Isolation Forest (to find novel anomalies) and a supervised LightGBM classifier (to learn known fraud patterns) for robust and accurate detection.
- **AI-Powered Reporting:** Uses the Gemini API to automatically generate a concise, actionable summary of the fraud findings, translating complex data into business intelligence.

---

## Project Structure

```

.
├── data/
│   └── criteo_sample.csv      # Sample of the Criteo dataset
├── ad_fraud_detection/
│   ├── __init__.py
│   ├── main.py                   # Main script to run the detection pipeline
│   ├── data_loader.py            # Module to load and preprocess data
│   ├── feature_engineering.py    # Module to create behavioral features
│   ├── graph_detector.py         # Module for graph-based anomaly detection
│   ├── ensemble_model.py         # Module for ensemble ML classification
│   └── report_generator.py       # Module for generating fraud reports with GenAI
├── .env                          # For storing your API key
├── requirements.txt              # Python dependencies
└── README.md                     # This file

````

---

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_name>
````

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file:**
   In the root directory of the project, create a file named `.env` and add your Gemini API key:

   ```bash
   GEMINI_API_KEY='your_actual_api_key_here'
   ```

5. **Download the data:**
   Place a sample of the Criteo Ad-Click Prediction dataset into the `data/` directory. The project is configured to read `criteo_sample.csv.gz`.

---

## How to Run

Execute the main pipeline script from the root directory of the project:

```bash
python -m ad_fraud_detection.main
```

---

## Sample Output

Below is a realistic example of the output you can expect when running the pipeline on a larger dataset (e.g., 50,000 rows).

```
Starting Digital Ad Fraud Detection Pipeline...

[Step 1/5] Loading data from data/criteo_sample.csv.gz...
Simulating realistic clickstream columns (IP, device, timestamp)...

Injecting synthetic fraud patterns (10% of data)...
Successfully simulated 5000 fraudulent clicks from 5 bot IPs.
Loaded 50000 records.

[Step 2/5] Engineering session-level and behavioral features...
Starting feature engineering...
... (progress bars) ...
Feature engineering complete.

[Step 3/5] Building graph and detecting anomalies...
Building IP-Device graph to find structural anomalies...
Graph created with 12534 nodes and 48750 edges.
Analyzing degrees for 1000 IP nodes...
Found 5 IPs exceeding the degree threshold of 10.
Identified 5 suspicious IPs/devices from graph analysis.

[Step 4/5] Applying ensemble model for fraud classification...

Running Isolation Forest for unsupervised anomaly detection...

Training supervised LightGBM model...
Creating heuristic labels for supervised model training...
Heuristically labeled 5450 out of 50000 records as fraud.

--- Supervised Model Performance ---
                precision    recall  f1-score   support

Legitimate (0)       0.99      1.00      1.00     13365
     Fraud (1)       0.98      0.93      0.95      1635

      accuracy                           0.99     15000
     macro avg       0.99      0.96      0.98     15000
  weighted avg       0.99      0.99      0.99     15000
------------------------------------

Applying supervised model to get fraud probabilities...
Final model flagged 1000 records as fraudulent.
Classification complete.

[Step 5/5] Generating automated fraud report...
--- Sending Prompt to Gemini API ---
--- Successfully received response from Gemini API ---

--- Weekly Fraud Report ---
**Subject: Weekly Ad Fraud Summary**

**1. Executive Summary**
This week, the system detected a coordinated and significant ad fraud operation. A total of **1,000 clicks** were flagged as fraudulent, primarily driven by a small botnet of 5 IP addresses. These bots exhibited clear signs of automation, including high-frequency clicking, device spoofing, and the use of non-standard user agents. Our graph analysis successfully identified all 5 core botnet IPs.

**2. Top Sources of Fraudulent Activity**
The fraudulent activity was highly concentrated. The following 5 IP addresses were responsible for over 95% of the flagged clicks and were all identified as structural anomalies in our graph analysis:
- IP: 10.2.112.58
- IP: 10.2.34.19
- IP: 10.2.201.130
- IP: 10.2.88.7
- IP: 10.2.145.211

**3. Key Fraud Patterns and Characteristics**
- **Botnet Behavior:** Each of the 5 IPs was connected to over 1,000 unique device IDs, a clear sign of a botnet attempting to appear as many different users.
- **Automated Tools:** All fraudulent clicks used `python-requests` or `curl` user-agents, indicating scripted attacks.
- **Time-Based Concentration:** The attacks were focused in a tight window between 3:00 and 3:15 AM UTC, a common tactic to avoid detection during peak business hours.

**4. Actionable Recommendations**
- **Immediate Blocking:** The 5 identified botnet IPs should be permanently blacklisted at the network edge.
- **Campaign Review:** Marketing teams should analyze campaigns targeted by these IPs to exclude this activity from performance metrics and budget calculations.
- **Alert Refinement:** Security teams should consider creating alerts for any IP that generates more than 500 clicks in a single hour, as this was a key indicator of this attack.

--------------------------
Pipeline finished successfully.
```

---
````
## Modules Explained

* **main.py:** The central script that orchestrates the entire pipeline, calling each module in sequence.
* **data\_loader.py:** Handles loading the raw clickstream data and programmatically injects realistic, synthetic fraud patterns to create a robust test dataset.
* **feature\_engineering.py:** Creates rich, session-level features from the raw data, including timing, frequency, duplicates, and user-agent analysis.
* **graph\_detector.py:** Constructs a graph of IP-to-device relationships and uses graph algorithms to find suspicious clusters and high-degree nodes indicative of botnets.
* **ensemble\_model.py:** Implements a hybrid of supervised (LightGBM) and unsupervised (Isolation Forest) models to score and classify clicks, complete with performance evaluation.
* **report\_generator.py:** Takes the structured output from the detection modules and uses a generative AI model (Gemini) to create a human-readable summary report.


