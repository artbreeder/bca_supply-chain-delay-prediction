# # src/preprocessing.py
# import pandas as pd
# from utils.Cleaners import clean_weight_column,clean_freight_cost
# def clean_data(df):
#     """
#     Performs data cleaning:
#     - date parsing
#     - missing values
#     - drop columns with too many nulls
#     """
#     #dropping redundant column
#     df = df.drop(columns=["Delivery Recorded Date","Molecule/Test Type","Item Description","Dosage","Dosage Form","Managed By"], errors="ignore")
#     # Convert dates
#     date_cols = [
#         "PQ First Sent to Client Date",
#         "PO Sent to Vendor Date",
#         "Scheduled Delivery Date",
#         "Delivered to Client Date",
#     ]
#     for col in date_cols:
#         df[col] = pd.to_datetime(df[col], errors='coerce')

#     # Drop rows where Scheduled Delivery Date is missing
#     df = df.dropna(subset=["Scheduled Delivery Date"])
    
#     # Clean specific columns
#     df = clean_weight_column(df)
#     df = clean_freight_cost(df)

#     # Simple fill for numeric nulls
#     num_cols = df.select_dtypes(include="number").columns
#     df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#     # Fill categorical nulls
#     cat_cols = df.select_dtypes(include="object").columns
#     df[cat_cols] = df[cat_cols].fillna("Unknown")

#     return df

#  src/preprocessing.py
from utils.debugging_funcs import log_shape   
# src/preprocessing.py
import pandas as pd
from utils.Cleaners import clean_weight_column,clean_freight_cost
def clean_data(df):
    """
    Performs data cleaning:
    - date parsing
    - missing values
    - drop columns with too many nulls
    """
    #dropping redundant column
    df = df.drop(columns=["Delivery Recorded Date","Molecule/Test Type","Item Description","Dosage","Dosage Form","Managed By"], errors="ignore")
    log_shape(df, "After dropping redundant columns")
    # Convert dates
    date_cols = [
        "PQ First Sent to Client Date",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    log_shape(df, "After converting date columns")
    

    # # Drop rows where Scheduled Delivery Date is missing
    # df = df.dropna(subset=["Scheduled Delivery Date"])
    
    # Clean specific columns
    df = clean_weight_column(df)
    df = clean_freight_cost(df)
    log_shape(df, "After cleaning Weight and Freight Cost columns")
    
    # Simple fill for numeric nulls
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    log_shape(df, "After filling numeric nulls")


    # Fill categorical nulls
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    log_shape(df, "After filling categorical nulls")
    
    # Standardize specific categorical values
    df["Shipment Mode"] = df["Shipment Mode"].replace("Unknown", "Others")

    return df
