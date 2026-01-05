# app.py - Simple ML Prediction Form

import streamlit as st
import pandas as pd
from src.predictor import DelayPredictor

# Page config
st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="📦",
    layout="centered"
)

# Load ML model
@st.cache_resource
def load_model():
    try:
        return DelayPredictor(
            model_path='artifacts/model.pkl',
            preprocessor_path='artifacts/preprocessor.pkl'
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run 'python main.py' first to train the model")
        return None

predictor = load_model()

# Title
st.title("📦 Supply Chain Delay Predictor")
st.markdown("Enter shipment details to predict delivery delays")
st.markdown("---")

# Get available options from predictor
if predictor:
    categories = predictor.get_available_categories()
    
else:
    # Fallback options if model not loaded
    categories = {
        'Vendor': ['Aurobindo Unit III, India', 'Cipla Ltd', 'Mylan Laboratories'],
        'Country': ['Nigeria', 'USA', 'India', 'South Africa', 'Kenya'],
        'Shipment Mode': ['Air', 'Ocean', 'Truck'],
        'Vendor INCO Term': ['EXW', 'FCA', 'DDP', 'CIF'],
        'Manufacturing Site': ['Hetero Unit III Hyderabad IN', 'Cipla Mumbai IN'],
        'Product Group': ['ARV', 'HRDT', 'ACT'],
        'Sub Classification': ['Adult', 'Pediatric']
    }

# Form
with st.form("prediction_form"):
    st.subheader("📋 Shipment Details")
    
    # Categorical Features
    col1, col2 = st.columns(2)
    
    with col1:
        vendor = st.selectbox(
            "Vendor *",
            options=categories.get('Vendor', []),
            help="Select the vendor"
        )
        
        country = st.selectbox(
            "Destination Country *",
            options=categories.get('Country', []),
            help="Where is the shipment going?"
        )
        
        shipment_mode = st.selectbox(
            "Shipment Mode *",
            options=categories.get('Shipment Mode', []),
            help="Air, Ocean, or Truck"
        )
        
        product_group = st.selectbox(
            "Product Group *",
            options=categories.get('Product Group', []),
            help="Type of medical product"
        )
    
    with col2:
        manufacturing_site = st.selectbox(
            "Manufacturing Site *",
            options=categories.get('Manufacturing Site', []),
            help="Where was it manufactured?"
        )
        
        vendor_inco_term = st.selectbox(
            "Vendor INCO Term *",
            options=categories.get('Vendor INCO Term', []),
            help="International shipping terms"
        )
        
        sub_classification = st.selectbox(
            "Sub Classification *",
            options=categories.get('Sub Classification', []),
            help="Adult or Pediatric"
        )
    
    st.markdown("---")
    st.subheader("📊 Quantities & Pricing")
    
    col3, col4 = st.columns(2)
    
    with col3:
        line_item_quantity = st.number_input(
            "Line Item Quantity *",
            min_value=1,
            value=500,
            help="Number of units"
        )
        
        pack_price = st.number_input(
            "Pack Price (USD) *",
            min_value=0.01,
            value=24.01,
            step=0.01,
            format="%.2f",
            help="Price per pack"
        )
        
        unit_price = st.number_input(
            "Unit Price (USD) *",
            min_value=0.01,
            value=0.80,
            step=0.01,
            format="%.2f",
            help="Price per unit"
        )
        
        weight = st.number_input(
            "Weight (Kilograms) *",
            min_value=0.1,
            value=150.5,
            step=0.1,
            format="%.1f",
            help="Total weight in kg"
        )
    
    with col4:
        unit_of_measure = st.number_input(
            "Unit of Measure (Per Pack) *",
            min_value=1,
            value=30,
            help="Units per pack"
        )
        
        freight_cost = st.number_input(
            "Freight Cost (USD) *",
            min_value=0.0,
            value=850.0,
            step=10.0,
            format="%.2f",
            help="Shipping cost"
        )
        
        insurance = st.number_input(
            "Line Item Insurance (USD) *",
            min_value=0.0,
            value=120.0,
            step=1.0,
            format="%.2f",
            help="Insurance cost"
        )
    
    # Calculate line item value
    line_item_value = pack_price * line_item_quantity
    st.info(f"💰 Total Line Item Value: ${line_item_value:,.2f}")
    
    st.markdown("---")
    
    # Submit button
    submitted = st.form_submit_button(
        "🔮 Predict Delay",
        type="primary",
        use_container_width=True
    )

# Handle prediction
if submitted:
    if not predictor:
        st.error("⚠️ Model not loaded. Please train the model first.")
    else:
        # Prepare input
        input_data = {
            'Unit of Measure (Per Pack)': unit_of_measure,
            'Line Item Quantity': line_item_quantity,
            'Line Item Value': line_item_value,
            'Pack Price': pack_price,
            'Unit Price': unit_price,
            'Weight (Kilograms)': weight,
            'Freight Cost (USD)': freight_cost,
            'Line Item Insurance (USD)': insurance,
            'Shipment Mode': shipment_mode,
            'Country': country,
            'Vendor': vendor,
            'Vendor INCO Term': vendor_inco_term,
            'Manufacturing Site': manufacturing_site,
            'Product Group': product_group,
            'Sub Classification': sub_classification
        }
        
        # Make prediction
        with st.spinner('Analyzing shipment risk...'):
            try:
                result = predictor.predict(input_data)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Main prediction
                if result['prediction'] == 'Delayed':
                    st.error("⚠️ **SHIPMENT LIKELY TO BE DELAYED**")
                else:
                    st.success("✅ **SHIPMENT EXPECTED ON TIME**")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Delay Probability",
                        f"{result['delay_probability']*100:.1f}%"
                    )
                
                with col_b:
                    st.metric(
                        "On-Time Probability",
                        f"{result['on_time_probability']*100:.1f}%"
                    )
                
                with col_c:
                    risk_color = {
                        'Low Risk': '🟢',
                        'Medium Risk': '🟡',
                        'High Risk': '🔴'
                    }
                    st.metric(
                        "Risk Level",
                        f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}"
                    )
                
                # Risk factors
                st.markdown("---")
                st.subheader("📊 Top Risk Factors")
                
                for i, factor in enumerate(result['top_risk_factors'], 1):
                    col_x, col_y = st.columns([3, 1])
                    with col_x:
                        st.write(f"**{i}. {factor['factor']}**")
                    with col_y:
                        st.write(f"{factor['risk_percentage']:.1f}%")
                
                # Unseen categories warning
                unseen = [k for k, v in result['input_validation'].items() if v]
                if unseen:
                    st.markdown("---")
                    st.warning(f"⚠️ **Note:** The following values were not seen during training and are using average risk: {', '.join(unseen)}")
                
                # Show raw data (collapsible)
                with st.expander("🔍 View Detailed Input"):
                    st.json(input_data)
                
                with st.expander("📈 View Risk Breakdown"):
                    risk_df = pd.DataFrame([
                        {'Risk Factor': k.replace('_', ' ').title(), 'Risk Score': f"{v:.4f}"}
                        for k, v in result['risk_breakdown'].items()
                    ])
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check that all required fields are filled correctly.")
                with st.expander("Error Details"):
                    st.code(str(e))

# Footer
st.markdown("---")
st.caption("💡 **Tip:** Try different combinations to see how shipment mode, country, and vendor affect delay predictions.")
st.caption("📌 Fields marked with * are required")