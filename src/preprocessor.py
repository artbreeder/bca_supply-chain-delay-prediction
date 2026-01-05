# src/preprocessor.py

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


class SupplyChainPreprocessor:
    """
    Complete preprocessing pipeline that saves risk maps and scaler.
    
    This class bridges the gap between:
    - Training: You have categorical features + target → calculate risks → scale
    - Prediction: User has categorical features (no target) → lookup risks → scale
    """
    
    def __init__(self):
        self.risk_maps = {}           # Stores category → delay_rate mappings
        self.default_risks = {}       # Fallback for unseen categories
        self.scaler = StandardScaler()
        self.feature_columns = None   # Exact order of features for model
        self.is_fitted = False
        
    # this df: pd.DataFrame is a type hint, not strict typing — it documents intent and helps tools, but Python does not enforce it at runtime
    def fit(self, df_before_risks: pd.DataFrame, target_col: str = 'Delayed'):
        """
        Fit the preprocessor on training data BEFORE risk calculation.
        
        THIS IS THE KEY: We capture the transformation logic here.
        
        Args:
            df_before_risks: Your dataframe AFTER clean_data() but BEFORE 
                           add_risk_score() and scale_numeric_features()
            target_col: Target variable name
            
        Returns:
            df_processed: Fully processed dataframe (with risks + scaled)
        """
        print("="*70)
        print("FITTING PREPROCESSOR")
        print("="*70)
        
        df = df_before_risks.copy()
        
        # STEP 1: Calculate and save all risk maps
        print("\n[1/2] Creating risk maps...")
        self._fit_risk_maps(df, target_col)
        
        # STEP 2: Apply risk transformations and get the processed df
        print("\n[2/2] Applying transformations...")
        df_processed = self._transform_training_data(df)
        
        # STEP 3: Fit scaler on the features (excluding target)
        feature_cols = [col for col in df_processed.columns if col != target_col]
        self.feature_columns = feature_cols
        
        X = df_processed[feature_cols]
        self.scaler.fit(X) #This stores the scaler parameters inside the object self.scaler
        
        # STEP 4: Scale the features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),#data to be scaled
            columns=feature_cols,
            index=df_processed.index #we wanted the rows to align with original df
        )
        
        # Add target back
        df_final = X_scaled.copy()
        df_final[target_col] = df_processed[target_col] #adding target column back and index aligns due to previous step
        
        self.is_fitted = True
        
        print("\n✓ Preprocessor fitted successfully!")
        print(f"  - Risk maps created: {len(self.risk_maps)}")
        print(f"  - Feature columns: {len(self.feature_columns)}")
        
        return df_final
    
    def _fit_risk_maps(self, df: pd.DataFrame, target_col: str):
        """
        Calculate all risk maps using your exact logic from feature_engineering.py
        """
        # min_samples = 30
        #reduced to 10 to have more categories covered in risk maps
        # min_samples = 10
        # further reduced to 5 to cover more categories
        min_samples = 5
        global_mean = df[target_col].mean()
        
        # 1. Joint risk: Shipment Mode + Country
        self._create_joint_risk_map(
            df, "Shipment Mode", "Country", target_col, min_samples
        )
        
        # 2. Joint risk: Vendor + Vendor INCO Term
        self._create_joint_risk_map(
            df, "Vendor", "Vendor INCO Term", target_col, min_samples
        )
        
        # 3. Individual risks
        for col in ["Manufacturing Site", "Vendor", "Product Group"]:
            self._create_single_risk_map(df, col, target_col, min_samples)
        
        # 4. Sub Classification risk
        self._create_single_risk_map(df, "Sub Classification", target_col, min_samples)
    
    def _create_joint_risk_map(self, df, col1, col2, target, min_samples):
        """Create risk map for two correlated features."""
        risk_name = f"{col1}_{col2}_risk"
        
        stats = (
            df.groupby([col1, col2])[target]
            .agg(["mean", "count"])
            .reset_index()
        )
        
        reliable = stats[stats["count"] >= min_samples]
        
        risk_map = {
            (row[col1], row[col2]): row["mean"]
            for _, row in reliable.iterrows()
        }
        
        global_mean = df[target].mean()
        
        self.risk_maps[risk_name] = risk_map
        self.default_risks[risk_name] = global_mean
        
        print(f"  ✓ {risk_name}: {len(risk_map)} combinations")
    
    def _create_single_risk_map(self, df, col, target, min_samples):
        """Create risk map for single feature."""
        risk_name = f"{col}_risk"
        
        stats = df.groupby(col)[target].agg(["mean", "count"])
        valid = stats["count"] >= min_samples
        
        risk_map = stats.loc[valid, "mean"].to_dict()
        global_mean = df[target].mean()
        
        self.risk_maps[risk_name] = risk_map
        self.default_risks[risk_name] = global_mean
        
        print(f"  ✓ {risk_name}: {len(risk_map)} categories")
    
    def _transform_training_data(self, df):
        """
        Apply all transformations to training data.
        This mimics your feature_engineering pipeline.
        """
        df_out = df.copy()
        
        # Add all risk features
        for risk_name, risk_map in self.risk_maps.items():
            default = self.default_risks[risk_name]
            
            if "_" in risk_name and risk_name.count("_risk") == 1:
                # Joint risk (e.g., "Shipment Mode_Country_risk")
                parts = risk_name.replace("_risk", "").split("_")
                if len(parts) >= 2:
                    col1, col2 = "_".join(parts[:-1]), parts[-1]
                    df_out[risk_name] = df_out.apply(
                        lambda row: risk_map.get((row[col1], row[col2]), default),
                        axis=1
                    )
            else:
                # Single feature risk
                source_col = risk_name.replace("_risk", "")
                df_out[risk_name] = df_out[source_col].map(risk_map).fillna(default)
        
        # Drop categorical columns (matching your finalize_features logic)
        drop_cols = [
            "Sub Classification", "Manufacturing Site", "Country", 
            "Shipment Mode", "Vendor", "Product Group", "Brand", 
            "First Line Designation", "PQ First Sent to Client Date",
            "PO Sent to Vendor Date", "Vendor INCO Term",
            # Additional from finalize_features
            "Manufacturer", "Item Description", "Molecule/Test Type",
            "ID", "PQ #", "PO / SO #", "ASN/DN #", "Item Code", "Project Code"
        ]
        
        df_out = df_out.drop(columns=drop_cols, errors='ignore')
        
        return df_out
    
    def transform(self, user_input: Dict[str, Any]) -> np.ndarray:
        """
        Transform raw user input to model-ready scaled features.
        
        THIS IS WHAT YOU'LL USE FOR PREDICTIONS.
        
        Args:
            user_input: Dictionary with raw categorical + numerical features
                {
                    'Unit of Measure (Per Pack)': 30,
                    'Line Item Quantity': 500,
                    'Shipment Mode': 'Air',
                    'Country': 'Nigeria',
                    'Vendor': 'Some Vendor Name',
                    ...
                }
        
        Returns:
            X_scaled: Scaled feature array ready for model.predict()
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform!")
        
        # Step 1: Create a copy with risk scores added
        transformed = user_input.copy()
        
        # Step 2: Calculate risk scores
        for risk_name in self.risk_maps.keys():
            transformed[risk_name] = self._calculate_risk_for_input(
                risk_name, user_input
            )
        
        # Step 3: Create DataFrame with exact feature order
        df = pd.DataFrame([transformed])
        
        # Ensure we have all required features
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only the features used in training
        df = df[self.feature_columns]
        
        # Step 4: Scale using fitted scaler
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def _calculate_risk_for_input(self, risk_name: str, user_input: Dict) -> float:
        """
        Look up the risk score for user's input.
        Returns default if category not seen during training.
        """
        risk_map = self.risk_maps[risk_name]
        default = self.default_risks[risk_name]
        
        # Joint risks
        if risk_name == "Shipment Mode_Country_risk":
            key = (user_input.get("Shipment Mode"), user_input.get("Country"))
            return risk_map.get(key, default)
        
        elif risk_name == "Vendor_Vendor INCO Term_risk":
            key = (user_input.get("Vendor"), user_input.get("Vendor INCO Term"))
            return risk_map.get(key, default)
        
        # Single feature risks
        elif risk_name == "Manufacturing Site_risk":
            return risk_map.get(user_input.get("Manufacturing Site"), default)
        
        elif risk_name == "Vendor_risk":
            return risk_map.get(user_input.get("Vendor"), default)
        
        elif risk_name == "Product Group_risk":
            return risk_map.get(user_input.get("Product Group"), default)
        
        elif risk_name == "Sub Classification_risk":
            return risk_map.get(user_input.get("Sub Classification"), default)
        
        else:
            return default
    
    def save(self, filepath: str = 'artifacts/preprocessor.pkl'):
        """Save the preprocessor for production use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str = 'artifacts/preprocessor.pkl'):
        """Load a saved preprocessor."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# HELPER FUNCTION FOR YOUR MAIN.PY
# ============================================================================

def create_and_save_preprocessor(df_clean: pd.DataFrame, target_col: str = "Delayed"):
    """
    This function should be called in your main.py AFTER clean_data()
    but BEFORE any risk calculations.
    
    Usage in main.py:
        df = clean_data(df)
        preprocessor = create_and_save_preprocessor(df)  # NEW LINE
        # ... continue with your existing feature engineering
    """
    preprocessor = SupplyChainPreprocessor()
    df_processed = preprocessor.fit(df_clean, target_col)
    
    # Save for later use in predictions
    import os
    os.makedirs('artifacts', exist_ok=True)
    preprocessor.save('artifacts/preprocessor.pkl')
    
    return preprocessor, df_processed