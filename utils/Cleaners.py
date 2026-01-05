import pandas as pd
# def clean_weight_column(df):
#     """
#     Cleans Weight (Kilograms) column:
#     - Forces numeric
#     - Removes invalid strings
#     - Drops or imputes bad values
#     """
#     df = df.copy()

#     df["Weight (Kilograms)"] = pd.to_numeric(
#         df["Weight (Kilograms)"],
#         errors="coerce"
#     )

#     # Remove zero or negative weights
#     df = df[df["Weight (Kilograms)"] > 0]

#     return df

# utils/Cleaners.py
import pandas as pd
import numpy as np

def clean_weight_column(df):
    df = df.copy()

    # Convert to numeric, invalid values → NaN
    df["Weight (Kilograms)"] = pd.to_numeric(
        df["Weight (Kilograms)"],
        errors="coerce"
    )

    # Replace negative or zero weights with NaN (do NOT drop rows)
    df.loc[df["Weight (Kilograms)"] <= 0, "Weight (Kilograms)"] = np.nan

    return df

def clean_freight_cost(df):
    """
    Cleans Freight Cost (USD) column by:
    - Converting non-numeric values to NaN
    - Replacing NaN with mean
    """
    df = df.copy()

    # Convert to numeric, force errors to NaN
    df["Freight Cost (USD)"] = pd.to_numeric(
        df["Freight Cost (USD)"],
        errors="coerce"
    )

    # Fill missing with mean
    mean_freight = df["Freight Cost (USD)"].mean()
    df["Freight Cost (USD)"] = df["Freight Cost (USD)"].fillna(mean_freight)

    return df

