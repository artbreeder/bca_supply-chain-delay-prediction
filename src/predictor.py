# src/predictor.py

import pickle
import pandas as pd
from src.preprocessor import SupplyChainPreprocessor
from typing import Dict, Any, List


class DelayPredictor:
    """
    Production-ready delay prediction system.
    
    This class:
    1. Loads the saved preprocessor (with risk maps)
    2. Loads the trained model
    3. Accepts raw user input
    4. Returns predictions + risk analysis
    """
    
    def __init__(self, 
                 model_path: str = 'artifacts/model.pkl',
                 preprocessor_path: str = 'artifacts/preprocessor.pkl'):
        
        print("Loading prediction artifacts...")
        
        # Load preprocessor
        self.preprocessor = SupplyChainPreprocessor.load(preprocessor_path)
        print(f"✓ Preprocessor loaded with {len(self.preprocessor.risk_maps)} risk maps")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded")
    
    def predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict delay for a single shipment.
        
        Args:
            user_input: Dictionary with raw values
                {
                    'Unit of Measure (Per Pack)': 30,
                    'Line Item Quantity': 500,
                    'Shipment Mode': 'Air',
                    'Country': 'Nigeria',
                    'Vendor': 'Some Vendor',
                    'Vendor INCO Term': 'EXW',
                    'Manufacturing Site': 'Some Site',
                    'Product Group': 'ARV',
                    'Sub Classification': 'Adult',
                    ... (all numerical features)
                }
        
        Returns:
            Dictionary with prediction and analysis
        """
        # Validate required fields
        self._validate_input(user_input)
        
        # Transform to model features
        X_scaled = self.preprocessor.transform(user_input)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Calculate risk breakdown
        risk_breakdown = {}
        for risk_name in self.preprocessor.risk_maps.keys():
            risk_score = self.preprocessor._calculate_risk_for_input(
                risk_name, user_input
            )
            risk_breakdown[risk_name] = risk_score
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'coef_'):
            feature_importance = self._get_top_contributing_features(
                X_scaled, self.model.coef_[0]
            )
        
        return {
            'prediction': 'Delayed' if prediction == 1 else 'On Time',
            'prediction_code': int(prediction),
            'delay_probability': float(probabilities[1]),
            'on_time_probability': float(probabilities[0]),
            'risk_breakdown': risk_breakdown,
            'risk_level': self._categorize_risk(probabilities[1]),
            'top_risk_factors': self._get_top_risks(risk_breakdown),
            'feature_importance': feature_importance,
            'input_validation': self._check_unseen_categories(user_input)
        }
    
    def predict_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict delays for multiple shipments.
        
        Args:
            inputs_list: List of input dictionaries
        
        Returns:
            List of prediction results
        """
        results = []
        for i, user_input in enumerate(inputs_list):
            try:
                result = self.predict(user_input)
                result['shipment_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'shipment_id': i,
                    'error': str(e),
                    'prediction': 'Error'
                })
        
        return results
    
    def _validate_input(self, user_input: Dict[str, Any]):
        """Validate that all required fields are present."""
        required_numerical = [
            'Unit of Measure (Per Pack)', 'Line Item Quantity', 
            'Line Item Value', 'Pack Price', 'Unit Price',
            'Weight (Kilograms)', 'Freight Cost (USD)', 
            'Line Item Insurance (USD)'
        ]
        
        required_categorical = [
            'Shipment Mode', 'Country', 'Vendor', 'Vendor INCO Term',
            'Manufacturing Site', 'Product Group', 'Sub Classification'
        ]
        
        missing = []
        for field in required_numerical + required_categorical:
            if field not in user_input or user_input[field] is None:
                missing.append(field)
        
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
    
    def _check_unseen_categories(self, user_input: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check which categorical values are unseen (not in training data).
        These will use default risk scores.
        """
        unseen = {}
        
        # Check each categorical feature
        checks = {
            'Vendor': ('Vendor_risk', user_input.get('Vendor')),
            'Country': ('Shipment Mode_Country_risk', 
                       (user_input.get('Shipment Mode'), user_input.get('Country'))),
            'Manufacturing Site': ('Manufacturing Site_risk', 
                                  user_input.get('Manufacturing Site')),
            'Product Group': ('Product Group_risk', user_input.get('Product Group')),
            'Sub Classification': ('Sub Classification_risk', 
                                  user_input.get('Sub Classification'))
        }
        
        for field_name, (risk_name, key) in checks.items():
            risk_map = self.preprocessor.risk_maps.get(risk_name, {})
            unseen[field_name] = key not in risk_map
        
        return unseen
    
    def _categorize_risk(self, delay_prob: float) -> str:
        """Categorize risk level based on probability."""
        if delay_prob < 0.3:
            return "Low Risk"
        elif delay_prob < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _get_top_risks(self, risk_breakdown: Dict[str, float], top_n: int = 3) -> List[Dict]:
        """Get the top risk factors sorted by risk score."""
        sorted_risks = sorted(
            risk_breakdown.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return [
            {
                'factor': risk_name.replace('_', ' ').replace(' risk', '').title(),
                'risk_score': risk_score,
                'risk_percentage': risk_score * 100
            }
            for risk_name, risk_score in sorted_risks
        ]
    
    def _get_top_contributing_features(self, X_scaled, coefficients, top_n: int = 5):
        """Get features contributing most to the prediction."""
        # Calculate contribution (feature_value * coefficient)
        contributions = X_scaled[0] * coefficients
        
        # Get top positive and negative
        feature_names = self.preprocessor.feature_columns
        
        contrib_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        return contrib_df.head(top_n).to_dict('records')
    
    def get_available_categories(self) -> Dict[str, List[str]]:
        """
        Get list of all categories seen during training.
        Useful for building dropdown menus in UI.
        """
        categories = {}
        
        # Single feature categories
        for risk_name, risk_map in self.preprocessor.risk_maps.items():
            if 'Vendor_risk' == risk_name:
                categories['Vendor'] = list(risk_map.keys())
            elif 'Manufacturing Site_risk' == risk_name:
                categories['Manufacturing Site'] = list(risk_map.keys())
            elif 'Product Group_risk' == risk_name:
                categories['Product Group'] = list(risk_map.keys())
            elif 'Sub Classification_risk' == risk_name:
                categories['Sub Classification'] = list(risk_map.keys())
        
        # Joint features - extract unique values
        if 'Shipment Mode_Country_risk' in self.preprocessor.risk_maps:
            modes_countries = list(self.preprocessor.risk_maps['Shipment Mode_Country_risk'].keys())
            categories['Shipment Mode'] = list(set([m for m, c in modes_countries]))
            categories['Country'] = list(set([c for m, c in modes_countries]))
        
        if 'Vendor_Vendor INCO Term_risk' in self.preprocessor.risk_maps:
            vendor_terms = list(self.preprocessor.risk_maps['Vendor_Vendor INCO Term_risk'].keys())
            categories['Vendor INCO Term'] = list(set([t for v, t in vendor_terms]))
        
        return categories


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DELAY PREDICTOR - USAGE EXAMPLE")
    print("="*70)
    
    # Initialize predictor
    predictor = DelayPredictor()
    
    # Example shipment
    test_shipment = {
        'Unit of Measure (Per Pack)': 30,
        'Line Item Quantity': 500,
        'Line Item Value': 12000.50,
        'Pack Price': 24.01,
        'Unit Price': 0.80,
        'Weight (Kilograms)': 150.5,
        'Freight Cost (USD)': 850.00,
        'Line Item Insurance (USD)': 120.00,
        'Shipment Mode': 'Air',
        'Country': 'Nigeria',
        'Vendor': 'Aurobindo Unit III, India',
        'Vendor INCO Term': 'EXW',
        'Manufacturing Site': 'Hetero Unit III Hyderabad IN',
        'Product Group': 'ARV',
        'Sub Classification': 'Adult'
    }
    
    # Make prediction
    result = predictor.predict(test_shipment)
    
    # Display results
    print(f"\n🎯 PREDICTION: {result['prediction']}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Delay Probability: {result['delay_probability']:.1%}")
    
    print(f"\n📊 TOP RISK FACTORS:")
    for i, risk in enumerate(result['top_risk_factors'], 1):
        print(f"   {i}. {risk['factor']}: {risk['risk_percentage']:.1f}%")
    
    print(f"\n⚠️  UNSEEN CATEGORIES:")
    for field, is_unseen in result['input_validation'].items():
        if is_unseen:
            print(f"   - {field} (using default risk)")
    
    print("\n" + "="*70)
    
    # Show available categories
    print("\n📋 AVAILABLE CATEGORIES FOR DROPDOWNS:")
    categories = predictor.get_available_categories()
    for category, values in categories.items():
        print(f"\n{category}: {len(values)} options")
        print(f"  Sample: {values[:3]}")