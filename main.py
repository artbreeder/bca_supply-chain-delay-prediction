# main.py
from src.data_loader import load_supply_chain_data
from src.preprocessing import clean_data
from src.feature_engineering import move_target_to_last,add_subclassification_risk, add_risk_score, create_delay_target, create_time_features, finalize_features,save_processed_data, scale_numeric_features,add_joint_risk_score,drop_features
from src.visualization import visualize_feature_correlation_heatmap,visualize_delay_by_first_line_designation,visualize_vendor_product_group_delay_heatmap,visualize_delay_counts, visualize_shipment_mode, visualize_cost_vs_delay,visualize_delay_by_shipment_mode,visualize_delay_proportion_by_shipment_mode,visualize_top_countries_delay_by_shipment_mode, delay_rate_by_weight_and_mode,delay_percentage_by_manufacturing_site,delay_percentage_by_fulfill_via,visualize_vendor_fulfill_delay_percentage
from utils.Analyze import analyze_vendor_inco_delay,analyze_vendor_inco_dependency,analyze_first_line_designation,delay_proportion_by_subclassification,check_delay_counts_by_mode_and_country,count_vendors_by_fulfill_via,vendor_fulfill_via_delay_counts,analyze_delay_by_weight,missing_values_percentage_in_PQ_to_PO_days
from utils.debugging_funcs import log_shape
from src.models.logistic_regression import train_logistic_regression, show_feature_importance
from src.evaluation.metrics import evaluate_model

def run_pipeline():
    # 1. Load & Clean
    # log_shape(df,"Loading raw data...")
    df = load_supply_chain_data("C:\\Users\\jesme\\OneDrive\\Desktop\\Supply_chain_ml_project\\data\\Raw\\SCMS_Delivery_History_Dataset_20150929.csv")
    
    log_shape(df,"Cleaning data...")
    df = clean_data(df)

    # 2. Feature Engineering
    log_shape(df,"Creating delay target...")
    df = create_delay_target(df)
    # log_shape(df,"Creating time features...")
    # df = create_time_features(df)
   # 3. Visualizations
    visualize_delay_counts(df)
    visualize_shipment_mode(df)
    visualize_cost_vs_delay(df)
    visualize_delay_by_shipment_mode(df)
    # #Here we got the real view of data that ocean mode has the highest proportion of delays
    visualize_delay_proportion_by_shipment_mode(df)

    visualize_top_countries_delay_by_shipment_mode(df, top_n=15)

    # #We enforced a minimum shipment threshold to avoid small-sample bias and ensure delay percentages reflected consistent operational behavior rather than random variation
    visualize_vendor_product_group_delay_heatmap(df, min_shipments=20)

    delay_percentage_by_fulfill_via(df)
    visualize_vendor_fulfill_delay_percentage(df)

    #we can see that first line designation is making some impact on delay percentage so we can keep this feature
    visualize_delay_by_first_line_designation(df)

    # Additional Analysis

    #Through this function we got to know that only 1 vendor is using rdd as fulfill via method and that heps us to prevent overfittig as if we only consider full via than there is 17.15 percent delay rate but in reality its only one vendor causing this high delay rate
    count_vendors_by_fulfill_via(df)

    #So this shows fulfill via method is not making any meaningfull impact on our data and also this is a classic issue of confounding variable where vendor and fulfill via are confounded. Hence we should drop fulfill via column
    vendor_fulfill_via_delay_counts(df)
    #dropping fulfill via column as it is not making any meaningfull impact on delay prediction and also it is confounded with vendor
    log_shape(df,"Dropping Fulfill Via column...")
    df = drop_features(df, columns=["Fulfill Via"])

    #checking missing values percentage in PQ_to_PO_days column
    # missing_values_percentage_in_PQ_to_PO_days(df)
    #we can drop this column as it has more than 60 percent missing values
    # df = drop_features(df, columns=["PQ_to_PO_days"])
    delay_rate_by_weight_and_mode(df) #this shows that weight doesnt making any impact on delay rate Hence weight and shipment mode are not correlated, Delay is not strongly driven by weight + shipment mode together

    # Does the manufacturing site (Manufacturer / Country of Origin) affect delay percentage?
    #analysis shows that a small subset of manufacturers consistently exhibits higher delay percentages compared to others. This suggests manufacturer-specific operational or logistical factors may contribute to delivery risk. These insights can be encoded as manufacturer-level risk features for predictive modeling
    delay_percentage_by_manufacturing_site(df)

    analyze_delay_by_weight(df, q=4)
    analyze_first_line_designation(df)
    print("-----------    new  info -------------------------")
    # analyze_vendor_inco_dependency(df)
    analyze_vendor_inco_delay(df)
    print("-----------    new  info -------------------------")
    print("------------------------------------")
    delay_proportion_by_subclassification(df)
    print("------------------------------------")
    # Risk scores
    #these two features are correlated for delay so we can create a composite risk score based on these features [shipment mode and country]
    # df = add_risk_score(df, "Shipment Mode")
    # df = add_risk_score(df, "Country")
    #To handle multicollinearity we can combine these two features into a single composite risk score and it would tell us that historically, how risky is this shipment mode when used in this country?
    log_shape(df,"Adding joint risk score for Shipment Mode and Country...")
    

    df = add_joint_risk_score(df, "Shipment Mode", "Country")
    
    df = add_joint_risk_score(df, "Vendor", "Vendor INCO Term")

    # Individual risk scores
    log_shape(df,"Adding risk scores for Manufacturing Site, Vendor, and Product Group...")
    df = add_risk_score(df, "Manufacturing Site")
    df = add_risk_score(df, "Vendor")
    df = add_risk_score(df, "Product Group")

    # Subclassification risk
    log_shape(df,"Adding Subclassification risk score...")
    df = add_subclassification_risk(df)

    # Drop unneeded features
    log_shape(df,"Dropping unneeded features...")
    df = drop_features(df, columns=["Sub Classification","Manufacturing Site","Country","Shipment Mode","Vendor","Product Group","Brand","First Line Designation","PQ First Sent to Client Date","PO Sent to Vendor Date","Vendor INCO Term"])
    
   

    # Finalize
    log_shape(df,"Finalizing features...")
    df = finalize_features(df)
    log_shape(df,"Scaling numeric features...")
    df = scale_numeric_features(df)


    # Save cleaned CSV
    # df.to_csv("data/processed/supply_chain_clean.csv", index=False)
    df = move_target_to_last(df, target_col="Delayed")
    visualize_feature_correlation_heatmap(df)
    log_shape(df,"Saving processed data...")
    save_processed_data(df, r"data/processed/supply_chain_final.csv")

    df_new = load_supply_chain_data("C:\\Users\\jesme\\OneDrive\\Desktop\\Supply_chain_ml_project\\data\\Processed\\supply_chain_final.csv")

    print("New df shape: ",df_new.shape)
    # print("Feature matrix shape:", X.shape)
        # Train model
    model, X_train, X_test, y_train, y_test = train_logistic_regression(df)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Feature importance
    show_feature_importance(model, X_train.columns)

if __name__ == "__main__":
    run_pipeline()
