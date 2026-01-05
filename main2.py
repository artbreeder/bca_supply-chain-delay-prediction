# main.py - MODIFIED VERSION
from src.data_loader import load_supply_chain_data
from src.preprocessing import clean_data
from src.feature_engineering import (
    move_target_to_last, add_subclassification_risk, add_risk_score, 
    create_delay_target, create_time_features, finalize_features,
    save_processed_data, scale_numeric_features, add_joint_risk_score, drop_features
)
from src.visualization import (
    visualize_feature_correlation_heatmap,
    visualize_delay_by_first_line_designation,
    visualize_vendor_product_group_delay_heatmap,
    visualize_delay_counts, visualize_shipment_mode, visualize_cost_vs_delay,
    visualize_delay_by_shipment_mode,
    visualize_delay_proportion_by_shipment_mode,
    visualize_top_countries_delay_by_shipment_mode, 
    delay_rate_by_weight_and_mode,
    delay_percentage_by_manufacturing_site,
    delay_percentage_by_fulfill_via,
    visualize_vendor_fulfill_delay_percentage
)
from utils.Analyze import (
    analyze_vendor_inco_delay, analyze_vendor_inco_dependency,
    analyze_first_line_designation, delay_proportion_by_subclassification,
    check_delay_counts_by_mode_and_country, count_vendors_by_fulfill_via,
    vendor_fulfill_via_delay_counts, analyze_delay_by_weight,
    missing_values_percentage_in_PQ_to_PO_days
)
from utils.debugging_funcs import log_shape
from src.models.logistic_regression import train_logistic_regression, show_feature_importance
from src.evaluation.metrics import evaluate_model

# NEW IMPORTS
from src.preprocessor import SupplyChainPreprocessor
import pickle
import os


def run_pipeline():
    """
    MODIFIED PIPELINE:
    1. Load & Clean
    2. Create preprocessor and save it (NEW!)
    3. Continue with existing feature engineering
    4. Train model and save it
    """
    
    # ========================================================================
    # STEP 1: Load & Clean
    # ========================================================================
    df = load_supply_chain_data(
        r"data/Raw/raw_data.csv"
    )
    
    log_shape(df, "Cleaning data...")
    df = clean_data(df)
    
    log_shape(df, "Creating delay target...")
    df = create_delay_target(df)
    
    # ========================================================================
    # STEP 2: CREATE AND SAVE PREPROCESSOR (NEW!)
    # ========================================================================
    # This captures the transformation logic BEFORE we do feature engineering
    print("\n" + "="*70)
    print("CREATING PREPROCESSOR FOR PRODUCTION USE")
    print("="*70)
    
    # Make sure artifacts directory exists
    os.makedirs('artifacts', exist_ok=True)
    
    # Create preprocessor from clean data
    preprocessor = SupplyChainPreprocessor()
    
    # Important: We need to save a copy of df BEFORE dropping Fulfill Via
    # because the preprocessor needs all categorical features
    df_for_preprocessor = df.copy()
    
    # Now continue with your analysis and feature drops
    # ========================================================================
    # STEP 3: Visualizations & Analysis (Your existing code)
    # ========================================================================
    visualize_delay_counts(df)
    visualize_shipment_mode(df)
    visualize_cost_vs_delay(df)
    visualize_delay_by_shipment_mode(df)
    visualize_delay_proportion_by_shipment_mode(df)
    visualize_top_countries_delay_by_shipment_mode(df, top_n=15)
    visualize_vendor_product_group_delay_heatmap(df, min_shipments=20)
    delay_percentage_by_fulfill_via(df)
    visualize_vendor_fulfill_delay_percentage(df)
    visualize_delay_by_first_line_designation(df)
    
    # Analysis
    count_vendors_by_fulfill_via(df)
    vendor_fulfill_via_delay_counts(df)
    
    # Drop Fulfill Via
    log_shape(df, "Dropping Fulfill Via column...")
    df = drop_features(df, columns=["Fulfill Via"])
    
    # Also drop from preprocessor's training data
    df_for_preprocessor = drop_features(df_for_preprocessor, columns=["Fulfill Via"])
    
    delay_rate_by_weight_and_mode(df)
    delay_percentage_by_manufacturing_site(df)
    analyze_delay_by_weight(df, q=4)
    analyze_first_line_designation(df)
    analyze_vendor_inco_delay(df)
    delay_proportion_by_subclassification(df)
    
    # ========================================================================
    # STEP 4: Now fit the preprocessor
    # ========================================================================
    print("\n" + "="*70)
    print("FITTING PREPROCESSOR ON TRAINING DATA")
    print("="*70)
    
    # Fit preprocessor (this will calculate risks and scale internally)
    df_processed = preprocessor.fit(df_for_preprocessor, target_col="Delayed")
    
    # Save the preprocessor
    preprocessor.save('artifacts/preprocessor.pkl')
    
    # ========================================================================
    # STEP 5: Continue with your original pipeline
    # ========================================================================
    # Note: We're using df_processed from preprocessor instead of manual steps
    
    # Visualize correlations on processed data
    visualize_feature_correlation_heatmap(df_processed)
    
    # Save processed data
    log_shape(df_processed, "Saving processed data...")
    save_processed_data(df_processed, r"data/processed/supply_chain_final.csv")
    
    # ========================================================================
    # STEP 6: Train model
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    model, X_train, X_test, y_train, y_test = train_logistic_regression(df_processed)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Feature importance
    show_feature_importance(model, X_train.columns)
    
    # Save model
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✓ Pipeline complete!")
    print(f"  - Preprocessor saved: artifacts/preprocessor.pkl")
    print(f"  - Model saved: artifacts/model.pkl")
    print(f"  - Processed data saved: data/processed/supply_chain_final.csv")


def test_prediction_pipeline():
    """
    Test function to verify that predictions work with raw input.
    Run this AFTER run_pipeline() completes.
    """
    print("\n" + "="*70)
    print("TESTING PREDICTION PIPELINE")
    print("="*70)
    
    # Load saved artifacts
    preprocessor = SupplyChainPreprocessor.load('artifacts/preprocessor.pkl')
    
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create a test input with RAW categorical values
    test_input = {
        # Numerical features (raw, not scaled)
        'Unit of Measure (Per Pack)': 30,
        'Line Item Quantity': 500,
        'Line Item Value': 12000.50,
        'Pack Price': 24.01,
        'Unit Price': 0.80,
        'Weight (Kilograms)': 150.5,
        'Freight Cost (USD)': 850.00,
        'Line Item Insurance (USD)': 120.00,
        
        # Categorical features (as strings)
        'Shipment Mode': 'Air',
        'Country': 'Nigeria',
        'Vendor': 'Aurobindo Unit III, India',
        'Vendor INCO Term': 'EXW',
        'Manufacturing Site': 'Hetero Unit III Hyderabad IN',
        'Product Group': 'ARV',
        'Sub Classification': 'Adult'
    }
    
    print("\n📦 Test Shipment Details:")
    print(f"  Quantity: {test_input['Line Item Quantity']}")
    print(f"  Vendor: {test_input['Vendor']}")
    print(f"  Country: {test_input['Country']}")
    print(f"  Shipment Mode: {test_input['Shipment Mode']}")
    
    # Transform to model features
    X_scaled = preprocessor.transform(test_input)
    
    print(f"\n🔄 Transformed to {X_scaled.shape[1]} scaled features")
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    print("\n🎯 Prediction Results:")
    print(f"  Prediction: {'DELAYED ⚠️' if prediction == 1 else 'ON TIME ✓'}")
    print(f"  Delay Probability: {probabilities[1]:.2%}")
    print(f"  On-Time Probability: {probabilities[0]:.2%}")
    
    # Show risk breakdown
    print("\n📊 Risk Breakdown:")
    for risk_name in preprocessor.risk_maps.keys():
        risk_score = preprocessor._calculate_risk_for_input(risk_name, test_input)
        print(f"  {risk_name}: {risk_score:.4f}")
    
    print("\n✓ Prediction pipeline working correctly!")


if __name__ == "__main__":
    # Run the full pipeline
    run_pipeline()
    
    # Test predictions
    print("\n" + "="*70)
    print("Would you like to test the prediction pipeline? (This will use saved artifacts)")
    print("="*70)
    test_prediction_pipeline()