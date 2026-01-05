# Supply Chain Delay Prediction

## 📌 Problem Statement
Analyze and predict shipment delays using real-world supply chain data.
The project focuses on reducing data leakage, avoiding volume bias, and
using risk-based feature engineering instead of naive encoding.

## 🔍 Key Insights
- Shipment Mode delays are volume-biased → normalized using risk scores
- Ocean shipments show higher delay probability in specific regions
- Vendor-specific behavior dominates Fulfill Via delays
- Weight shows monotonic increase in delay probability
- Manufacturer and Country are strong predictors

## 🧠 Feature Engineering
- Binary delay target
- Risk encoding for:
  - Shipment Mode × Country
  - Manufacturer
- Dropped high-missing and redundant features
- Numeric scaling only where appropriate

## 📊 Visualizations
- Delay proportions by shipment mode
- Vendor × Fulfill Via × Delay analysis
- Weight bucket vs delay percentage
- Country-level delay breakdowns

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

