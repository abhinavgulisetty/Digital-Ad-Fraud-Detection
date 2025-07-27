# ad_fraud_detection/graph_detector.py

import pandas as pd
import networkx as nx
from tqdm import tqdm

def find_graph_anomalies(df, ip_degree_threshold=10):
    """
    Builds a graph of IPs and Devices to find structural anomalies.

    This function creates a bipartite graph where one set of nodes is IPs and
    the other is devices. An edge exists if a click from an IP is associated
    with a device. It then identifies IPs connected to an unusually high number
    of devices, which is a strong indicator of a botnet or proxy farm.

    Args:
        df (pd.DataFrame): The feature-engineered DataFrame. It must contain
                           'ip' and 'device_id' columns.
        ip_degree_threshold (int): The number of unique devices an IP must be
                                   connected to before it's flagged as an anomaly.

    Returns:
        list: A list of IP addresses flagged as anomalous.
    """
    print("Building IP-Device graph to find structural anomalies...")

    # Create a graph from the pandas DataFrame.
    # We'll create a bipartite graph connecting IPs to Devices.
    G = nx.Graph()

    # To speed up graph creation, we'll iterate over unique IP-device pairs.
    ip_device_pairs = df[['ip', 'device_id']].drop_duplicates()

    # Add nodes and edges
    # We can add all nodes at once for efficiency
    all_ips = ip_device_pairs['ip'].unique()
    all_devices = ip_device_pairs['device_id'].unique()
    
    # Add nodes with bipartite attribute
    G.add_nodes_from(all_ips, bipartite=0) # 0 for IPs
    G.add_nodes_from(all_devices, bipartite=1) # 1 for Devices

    # Add edges
    print("Adding edges to the graph...")
    for _, row in tqdm(ip_device_pairs.iterrows(), total=ip_device_pairs.shape[0], desc="Building Graph"):
        G.add_edge(row['ip'], row['device_id'])

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Anomaly Detection using Graph Properties ---
    # We will identify IPs with an abnormally high degree (connected to many devices).
    anomalous_ips = []
    
    ip_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    
    print(f"Analyzing degrees for {len(ip_nodes)} IP nodes...")
    for ip in tqdm(ip_nodes, desc="Analyzing IP Degrees"):
        degree = G.degree(ip)
        if degree > ip_degree_threshold:
            anomalous_ips.append(ip)

    if anomalous_ips:
        print(f"Found {len(anomalous_ips)} IPs exceeding the degree threshold of {ip_degree_threshold}.")
    else:
        print("No IPs exceeded the degree threshold.")

    return anomalous_ips
