
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_delay_counts(df):
    """
    Shows how many shipments were delayed vs on-time.
    """
    sns.countplot(data=df, x="Delayed")
    plt.title("Count of Delayed vs On-Time Shipments")
    plt.xlabel("Delayed (1 = yes, 0 = no)")
    plt.show()

def visualize_shipment_mode(df):
    """
    Plot shipment mode frequency.
    """
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, y="Shipment Mode", order=df["Shipment Mode"].value_counts().index)
    plt.title("Shipment Mode Frequency")
    plt.show()

def visualize_cost_vs_delay(df):
    """
    Shows freight cost distribution by delay status.
    """
    sns.boxplot(data=df, x="Delayed", y="Freight Cost (USD)")
    plt.title("Freight Cost vs Delay")
    plt.show()

def visualize_delay_by_shipment_mode(df):
    """
    Shows delayed vs on-time shipments by shipment mode
    """

    """ Here i realized that we have some volume bias as there are more air shipments than other modes so its obvious we would get more delays so to remove this volume bias we would deal with proportions instead of counts
    but for now lets just visualize counts first"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a copy to avoid modifying original df
    plot_df = df.copy()

    # Convert Delayed to categorical labels for plotting
    plot_df["Delayed_Label"] = plot_df["Delayed"].map({
        0: "On-Time",
        1: "Delayed"
    })

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=plot_df,
        x="Shipment Mode",
        hue="Delayed_Label"
    )

    plt.title("Delayed vs On-Time Shipments by Shipment Mode")
    plt.xlabel("Shipment Mode")
    plt.ylabel("Number of Shipments")
    plt.xticks(rotation=30)
    plt.legend(title="Delivery Status")
    plt.tight_layout()
    plt.show()

def visualize_delay_proportion_by_shipment_mode(df):
    """
    Shows proportion of delayed vs on-time shipments
    for each shipment mode (volume-normalized)
    """
    import matplotlib.pyplot as plt

    # Copy to avoid side effects
    plot_df = df.copy()

    # Create readable labels
    plot_df["Delayed_Label"] = plot_df["Delayed"].map({
        0: "On-Time",
        1: "Delayed"
    })

    # Compute proportions safely
    proportion_df = (
        plot_df
        .groupby("Shipment Mode")["Delayed_Label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Plot stacked bar chart
    proportion_df.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    plt.title("Proportion of Delayed vs On-Time Shipments by Shipment Mode")
    plt.xlabel("Shipment Mode")
    plt.ylabel("Proportion")
    plt.xticks(rotation=30)
    plt.legend(title="Delivery Status")
    plt.tight_layout()
    plt.show()

def visualize_top_countries_delay_by_shipment_mode(df, top_n=10):
    """
    Heatmap of delay rate by shipment mode for top N countries
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get top N countries by shipment volume
    top_countries = (
        df["Country"]
        .value_counts()
        .head(top_n)
        .index
    )

    # Filter data
    filtered_df = df[df["Country"].isin(top_countries)]

    # Compute delay rate
    delay_rate_df = (
        filtered_df
        .groupby(["Country", "Shipment Mode"])["Delayed"]
        .mean()
        .reset_index()
    )

    # Pivot for heatmap
    pivot_df = delay_rate_df.pivot(
        index="Country",
        columns="Shipment Mode",
        values="Delayed"
    )

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="Reds"
    )

    plt.title("Delay Rate by Shipment Mode for Top 10 Countries")
    plt.xlabel("Shipment Mode")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()


def delay_rate_by_weight_and_mode(df):
    """
    Computes delay rate across weight buckets and shipment modes
    """
    import pandas as pd

    df = df.copy()

    # ✅ Convert weight to numeric safely
    df["Weight (Kilograms)"] = pd.to_numeric(
        df["Weight (Kilograms)"],
        errors="coerce"
    )

    # Remove invalid or zero weights
    df = df[df["Weight (Kilograms)"] > 0]

    # Create weight buckets
    df["Weight_Bucket"] = pd.qcut(
        df["Weight (Kilograms)"],
        q=4,
        labels=["Light", "Medium", "Heavy", "Very Heavy"]
    )

    # Compute delay rate
    delay_rate = (
        df
        .groupby(["Shipment Mode", "Weight_Bucket"])["Delayed"]
        .mean()
        .unstack()
        .mul(100)
        .round(2)
    )

    print("\nDelay rate by shipment mode and weight bucket:")
    print(delay_rate)

    return delay_rate

def delay_percentage_by_manufacturing_site(df, top_n=10):
    """
    Shows delay percentage by manufacturing site
    (only top N sites by shipment volume)
    """
    import matplotlib.pyplot as plt

    # Check column existence defensively
    if "Manufacturing Site" not in df.columns:
        raise KeyError(
            "Column 'Manufacturing Site' not found. "
            "Check df.columns for the correct name."
        )

    # Compute delay percentage
    delay_pct = (
        df
        .groupby("Manufacturing Site")["Delayed"]
        .mean()
        .mul(100)
    )

    # Keep only top N by volume (important to avoid noise)
    top_sites = (
        df["Manufacturing Site"]
        .value_counts()
        .head(top_n)
        .index
    )

    delay_pct = delay_pct.loc[top_sites].sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    delay_pct.plot(kind="bar")

    plt.title(f"Delay Percentage by Manufacturing Site (Top {top_n})")
    plt.ylabel("Delay Percentage (%)")
    plt.xlabel("Manufacturing Site")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return delay_pct

def delay_percentage_by_fulfill_via(df):
    delay_pct = (
        df.groupby("Fulfill Via")["Delayed"]
        .mean()
        .sort_values(ascending=False) * 100
    )

    print("\nDelay percentage by Fulfill Via:")
    print(delay_pct)
def visualize_vendor_fulfill_delay_percentage(df, top_n=15):
    """
    Visualize delay percentage by Vendor × Fulfill Via
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Copy for safety
    plot_df = df.copy()

    # Calculate delay percentage
    delay_pct = (
        plot_df
        .groupby(["Vendor", "Fulfill Via"])["Delayed"]
        .mean()
        .mul(100)
        .reset_index(name="Delay_Percentage")
    )

    # Keep only top N vendors by volume
    top_vendors = (
        plot_df["Vendor"]
        .value_counts()
        .head(top_n)
        .index
    )

    delay_pct = delay_pct[delay_pct["Vendor"].isin(top_vendors)]

    # Plot
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=delay_pct,
        x="Vendor",
        y="Delay_Percentage",
        hue="Fulfill Via"
    )

    plt.title("Delay Percentage by Vendor and Fulfill Via")
    plt.xlabel("Vendor")
    plt.ylabel("Delay Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Fulfill Via")
    plt.tight_layout()
    plt.show()

def visualize_vendor_product_group_delay_heatmap(df, min_shipments):
    """
    Heatmap of delay percentage for Vendor vs Product Group
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    summary = (
        df
        .groupby(["Vendor", "Product Group"])["Delayed"]
        .agg(
            total_shipments="count",
            delay_rate="mean"
        )
        .reset_index()
    )

    # Filter noise
    summary = summary[summary["total_shipments"] >= min_shipments]

    # Convert to percentage
    summary["delay_percentage"] = summary["delay_rate"] * 100

    pivot = summary.pivot(
        index="Vendor",
        columns="Product Group",
        values="delay_percentage"
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot,
        cmap="Reds",
        annot=True,
        fmt=".1f",
        linewidths=0.5
    )

    plt.title("Delay Percentage Heatmap: Vendor vs Product Group")
    plt.ylabel("Vendor")
    plt.xlabel("Product Group")
    plt.tight_layout()
    plt.show()

def visualize_delay_by_first_line_designation(df):
    """
    Bar plot of delay percentage by First Line Designation
    """
    import matplotlib.pyplot as plt

    summary = (
        df.groupby("First Line Designation")["Delayed"]
        .mean()
        * 100
    )

    plt.figure(figsize=(6, 4))
    summary.plot(kind="bar")
    plt.title("Delay Percentage by First Line Designation")
    plt.ylabel("Delay Percentage")
    plt.xlabel("First Line Designation")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
def visualize_feature_correlation_heatmap(df):
    """
    Plots correlation heatmap for numeric features
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Correlation matrix
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
