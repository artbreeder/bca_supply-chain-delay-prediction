def check_delay_counts_by_mode_and_country(
    df,
    shipment_mode="Ocean",
    top_n=15
):
    """
    Prints shipment counts and delay counts
    for a given shipment mode across top N countries
    """

    # Filter for the shipment mode
    filtered_df = df[df["Shipment Mode"] == shipment_mode]

    # Aggregate counts
    summary = (
        filtered_df
        .groupby("Country")
        .agg(
            total_shipments=("Delayed", "count"),
            delayed_shipments=("Delayed", "sum")
        )
        .sort_values("delayed_shipments", ascending=False)
        .head(top_n)
    )

    # Add delay rate
    summary["delay_rate"] = (
        summary["delayed_shipments"] / summary["total_shipments"]
    )

    print(f"\nDelay summary for shipment mode: {shipment_mode}")
    print(summary)

    return summary


def count_vendors_by_fulfill_via(df):
    """
    Counts how many unique vendors use each Fulfill Via method
    """

    vendor_counts = (
        df.groupby("Fulfill Via")["Vendor"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nNumber of unique vendors by Fulfill Via:")
    print(vendor_counts)

    return vendor_counts
def vendor_fulfill_via_delay_counts(df):
    """
    Counts vendors by Fulfill Via and their delay behavior
    """
    summary = (
        df
        .groupby(["Vendor", "Fulfill Via"])
        .agg(
            total_shipments=("Delayed", "count"),
            delayed_shipments=("Delayed", "sum"),
            delay_percentage=("Delayed", "mean")
        )
        .reset_index()
        .sort_values(by="delay_percentage", ascending=False)
    )

    # Convert to percentage
    summary["delay_percentage"] = summary["delay_percentage"] * 100

    print("\nVendor × Fulfill Via × Delay Summary:")
    print(summary)

    return summary

def analyze_delay_by_weight(df, q=4):
    """
    Analyzes whether weight impacts delay percentage
    """
    import pandas as pd

    df = df.copy()

    # Ensure weight is numeric
    df = df.dropna(subset=["Weight (Kilograms)"])

    # Create weight buckets
    df["Weight_Bucket"] = pd.qcut(
        df["Weight (Kilograms)"],
        q=4,
        labels=["Very Light", "Light", "Heavy", "Very Heavy"]
    )


    # Compute delay percentage
    summary = (
        df.groupby("Weight_Bucket")["Delayed"]
        .mean()
        .mul(100)
        .reset_index(name="Delay_Percentage")
    )

    print("\nDelay percentage by weight bucket:")
    print(summary)

    return summary

def missing_values_percentage_in_PQ_to_PO_days(df):
    total = len(df)
    missing = df["PQ_to_PO_days"].isna().sum()
    print(f"PQ_to_PO_days missing: {missing}/{total} ({missing/total:.2%})")

def delay_proportion_by_subclassification(df, min_samples=30):
    """
    Computes delay count and delay proportion per Sub Classification
    """

    summary = (
        df.groupby("Sub Classification")["Delayed"]
        .agg(
            total_shipments="count",
            delayed_shipments="sum",
            delay_proportion="mean"
        )
        .reset_index()
    )

    # Filter small groups to avoid noise
    summary = summary[summary["total_shipments"] >= min_samples]

    # Sort by delay proportion
    summary = summary.sort_values(
        by="delay_proportion", ascending=False
    )

    print("\nDelay proportion by Sub Classification:")
    print(summary)

    return summary
def analyze_first_line_designation(df):
    import pandas as pd

    summary = (
        df.groupby("First Line Designation")["Delayed"]
        .agg(
            total_shipments="count",
            delayed_shipments="sum",
            delay_percentage="mean"
        )
        .reset_index()
    )

    summary["delay_percentage"] *= 100
    print("\nFirst Line Designation Summary:")
    print(summary)

    return summary
def analyze_vendor_inco_dependency(df, top_n=20):
    """
    Shows how many INCO terms each vendor uses
    """
    import pandas as pd

    vendor_inco_counts = (
        df.groupby("Vendor")["Vendor INCO Term"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nNumber of unique INCO terms per vendor (top vendors):")
    print(vendor_inco_counts.head(top_n))

    return vendor_inco_counts

def analyze_vendor_inco_delay(df, top_n=10, min_shipments=30):
    """
    Analyze Vendor × Vendor INCO Term × Delay proportion
    """

    import pandas as pd

    # Count shipments per vendor
    vendor_counts = (
        df.groupby("Vendor")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # Filter to top vendors
    filtered_df = df[df["Vendor"].isin(vendor_counts)]

    summary = (
        filtered_df
        .groupby(["Vendor", "Vendor INCO Term"])
        .agg(
            total_shipments=("Delayed", "count"),
            delayed_shipments=("Delayed", "sum")
        )
        .reset_index()
    )

    # Compute delay proportion
    summary["delay_proportion"] = (
        summary["delayed_shipments"] / summary["total_shipments"]
    )

    # Filter small samples
    summary = summary[summary["total_shipments"] >= min_shipments]

    # Sort for readability
    summary = summary.sort_values(
        by="delay_proportion",
        ascending=False
    )

    print("\nVendor × Vendor INCO Term × Delay Proportion:")
    print(summary)

    return summary
