# src/data_loader.py
import pandas as pd

def load_supply_chain_data(file_path):
    """
    Loads the supply chain shipment pricing CSV file.
    """
    df = pd.read_csv(file_path,
                     encoding="ISO-8859-1")
    print(f"Loaded dataset with shape: {df.shape}")
    return df
