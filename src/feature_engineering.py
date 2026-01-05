# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


# def create_delay_target(df):
#     """
#     Creates binary target:
#     1 = delayed
#     0 = on-time
#     """
#     df["Delayed"] = (
#         df["Delivered to Client Date"] > df["Scheduled Delivery Date"]
#     ).astype(int)

#     # Drop leakage columns
#     df = df.drop(
#         columns=[
#             "Delivered to Client Date",
#             "Scheduled Delivery Date",
#             "Delivery Recorded Date"
#         ],
#         errors="ignore"
#     )

#     return df

# #PQ Date  ───── negotiation ─────>  PO Date
# """PQ is Purchase Quotient and PO is Purchase Order and we are trying to calculate a new variable that tells us how many days it took to finnaly placing the order 
# this new feature might help us understand delays better.
# """
# def create_time_features(df):
#     df["PQ_to_PO_days"] = (
#         df["PO Sent to Vendor Date"] - df["PQ First Sent to Client Date"]
#     ).dt.days

#     # Drop raw date columns
#     df = df.drop(
#         columns=[
#             "PQ First Sent to Client Date",
#             "PO Sent to Vendor Date"
#         ],
#         errors="ignore"
#     )

#     return df


# def preprocess_features(df):
#     """
#     Normalizes numeric features and encodes categorical features.
#     Returns:
#     X_processed, y, fitted_preprocessor
#     """

#     # Drop identifiers / high-cardinality noise
#     drop_cols = [
#         "ID",
#         "PQ #",
#         "PO / SO #",
#         "ASN/DN #",
#         "Item Code",
#         "Project Code"
#     ]
#     df = df.drop(columns=drop_cols, errors="ignore")

#     # Extract target
#     y = df["Delayed"]
#     X = df.drop(columns=["Delayed"])

#     # Feature separation
#     num_features = X.select_dtypes(include=["int64", "float64"]).columns
#     cat_features = X.select_dtypes(include=["object"]).columns

#     # Pipelines
#     numeric_transformer = Pipeline(
#         steps=[("scaler", StandardScaler())]
#     )

#     # categorical_transformer = Pipeline(
#     #     steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
#     # )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, num_features),
#             # ("cat", categorical_transformer, cat_features)
#         ]
#     )

#     X_processed = preprocessor.fit_transform(X)

#     return X_processed, y, preprocessor

# def save_processed_data(X_processed, y, preprocessor, output_path):
#     """
#     Saves processed features and target as a CSV with proper feature names
#     """

#     # Get numeric feature names
#     num_features = preprocessor.transformers_[0][2]

#     # Get categorical feature names after one-hot encoding
#     # cat_encoder = preprocessor.transformers_[1][1]["encoder"]
#     # cat_features = preprocessor.transformers_[1][2]
#     # cat_feature_names = cat_encoder.get_feature_names_out(cat_features)

#     # Combine all feature names
#     feature_names = list(num_features) #+ list(cat_feature_names)

#     # Create DataFrame
#     X_df = pd.DataFrame(X_processed, columns=feature_names)

#     # Add target
#     X_df["Delayed"] = y.values

#     # Save to disk
#     X_df.to_csv(output_path, index=False)

#     print(f"Processed dataset saved to: {output_path}")
#     """

# src/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_delay_target(df):
    """
    Creates binary target:
    1 = delayed
    0 = on-time
    """
    df = df.copy()

    df["Delayed"] = (
        df["Delivered to Client Date"] > df["Scheduled Delivery Date"]
    ).astype(int)

    # Drop leakage columns
    df.drop(
        columns=[
            "Delivered to Client Date",
            "Scheduled Delivery Date",
            "Delivery Recorded Date"
        ],
        inplace=True,
        errors="ignore"
    )

    return df


def create_time_features(df):
    """
    PQ → PO lead time feature
    """
    df = df.copy()

    df["PQ_to_PO_days"] = (
        df["PO Sent to Vendor Date"] - df["PQ First Sent to Client Date"]
    ).dt.days

    df.drop(
        columns=[
            "PQ First Sent to Client Date",
            "PO Sent to Vendor Date"
        ],
        inplace=True,
        errors="ignore"
    )

    return df

def drop_features(df, columns):
    """
    Drops specified columns from the dataframe
    """
    df = df.copy()
    df.drop(columns=columns, inplace=True, errors="ignore")
    print(f"Dropped columns: {columns}")
    return df


def add_risk_score(df, column_name, target="Delayed", min_samples=30):
    """
    Converts a categorical column into a delay risk score
    """
    df = df.copy()

    stats = (
        df.groupby(column_name)[target]
        .agg(["mean", "count"])
    )

    # Filter low-support categories
    valid = stats["count"] >= min_samples
    risk_map = stats.loc[valid, "mean"]

    global_mean = df[target].mean()

    df[f"{column_name}_risk"] = (
        df[column_name].map(risk_map).fillna(global_mean)
    )

    return df

def add_subclassification_risk(df, min_samples=30):
    """
    Adds delay risk score for Sub Classification
    """
    df = df.copy()

    stats = (
        df.groupby("Sub Classification")["Delayed"]
        .agg(["mean", "count"])
    )

    valid = stats["count"] >= min_samples
    risk_map = stats.loc[valid, "mean"]

    global_mean = df["Delayed"].mean()

    df["Sub_Classification_risk"] = (
        df["Sub Classification"]
        .map(risk_map)
        .fillna(global_mean)
    )

    return df

def finalize_features(df):
    """
    Drops raw categorical columns after risk encoding
    """
    drop_cols = [
        "Shipment Mode",
        "Country",
        "Manufacturer",
        "Item Description",
        "Molecule/Test Type",
        "ID",
        "PQ #",
        "PO / SO #",
        "ASN/DN #",
        "Item Code",
        "Project Code"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def scale_numeric_features(df, exclude_cols=["Delayed"]):
    """
    Scales numeric features only
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [c for c in num_cols if c not in exclude_cols]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def save_processed_data(df, output_path):
    """
    Saves final processed dataset
    """
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
def add_joint_risk_score(
    df,
    col_1,
    col_2,
    target="Delayed",
    min_samples=30
):
    """
    Creates a joint risk score for two correlated categorical features
    Example: Shipment Mode + Country
    """

    df = df.copy()

    # Compute joint stats
    stats = (
        df
        .groupby([col_1, col_2])[target]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Keep only reliable combinations
    reliable = stats[stats["count"] >= min_samples]

    # Create lookup dictionary
    joint_risk_map = {
        (row[col_1], row[col_2]): row["mean"]
        for _, row in reliable.iterrows()
    }

    global_mean = df[target].mean()

    # Assign risk score
    df[f"{col_1}_{col_2}_risk"] = df.apply(
        lambda row: joint_risk_map.get(
            (row[col_1], row[col_2]),
            global_mean
        ),
        axis=1
    )

    return df

def move_target_to_last(df, target_col="Delayed"):
    """
    Moves the target column to the last position
    """
    cols = [c for c in df.columns if c != target_col] + [target_col]
    return df[cols]
