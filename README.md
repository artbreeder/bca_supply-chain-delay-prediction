# Supply Chain Delay Prediction System
This project is a machine learning system designed to predict whether a supply chain order will be delayed or delivered on time. It uses historical shipment data to build a production-grade pipeline that prioritizes reproducibility, robustness, and explainability over black-box complexity.

# Project Goals
The focus of this repository is to demonstrate a disciplined and practical approach to machine learning, specifically:
Clean and modular data preprocessing
Feature engineering designed to prevent data leakage
Reproducible workflows that can be executed with a single command
Explainable predictions that provide clear business value

# The Core Concept
A key challenge in supply chain data is handling categorical variables such as Vendors, Manufacturing Sites, Shipment Modes, and Destination Countries.
In early iterations, standard one-hot encoding was used. However, this approach caused significant issues:
Hundreds of unique categories led to a very large feature space
The resulting dataset became sparse and memory-intensive
Training slowed down without providing proportional predictive gains
To address this, the project adopts risk-based encoding.
##Risk-Based Encoding
Instead of representing categories as binary vectors, each category is converted into a numerical risk score based on historical performance.
For example:
If a vendor has historically delayed 30% of shipments, the vendor risk is encoded as 0.30.
This approach:
Keeps the dataset compact
Reduces dimensionality
Improves training efficiency
Makes features directly interpretable as business risk indicators

# Technical Approach
1. Risk-Based Feature Engineering
Risk scores are calculated for multiple business dimensions, including:
Vendor risk
Manufacturing Site risk
Product Group risk
Joint risks such as:
Shipment Mode × Destination Country
Vendor × INCO Term
To ensure stability, minimum sample thresholds are applied so that rare categories do not produce unreliable or biased risk estimates.

2. Data Leakage Prevention
The system is built with a strict separation between training and inference:
All risk mappings are learned only during the training phase
These mappings are frozen and saved
During prediction, the system only performs lookups using the stored mappings
This design ensures that the model never has access to information it should not see, preserving honest evaluation and real-world reliability.

3. Model Selection
The model used is Logistic Regression.
While more complex models are available, Logistic Regression was chosen because it provides:
Stability across different data distributions
Strong performance with well-engineered features
Clear interpretability
The ability to inspect feature contributions
This makes it suitable for operational and decision-support use cases in supply chain management.

4. Explainability
The system does not output only a binary prediction. For each shipment, it provides:
The probability of delay
A categorized risk level (Low, Medium, High)
The primary factors contributing to that risk
This allows users to understand why a shipment is considered risky, rather than treating the model as a black box.
How to Run the Project Prerequisites
Python 3.9 or higher

# Installation and Execution
The project includes an automation script that handles virtual environment creation, dependency installation, model training, and application startup.

# Clone the repository
git clone https://github.com/Jesmeeksingh/supply-chain-delay-prediction.git cd supply-chain-delay-prediction

# Run the startup script
python start_project.py

This single command will:

Create a virtual environment if one does not exist

Install all required dependencies

Run the training and preprocessing pipeline

Launch the Streamlit application

Access the Application

Once the script finishes, the Streamlit interface will be available at:

http://localhost:8501

Streamlit usage prompts are disabled via project-level configuration, making the setup fully non-interactive.

# Design Decisions
Dimensionality Management
One-hot encoding was replaced with risk-based encoding to prevent feature explosion, reduce memory usage, and improve training efficiency.

# Automation
A Python-based startup script (start_project.py) is used instead of OS-specific shell scripts to ensure the project runs consistently across platforms.

# Artifact Management
Trained model files (.pkl) are intentionally excluded from version control. They are treated as reproducible outputs of the code rather than source artifacts.

# Simplicity and Trust
The project avoids opaque, black-box approaches. Every transformation and prediction step is designed to be understandable to technical and non-technical stakeholders alike.

# Summary
This project demonstrates how to build a practical, explainable, and reproducible machine learning system for supply chain delay prediction. It focuses on engineering discipline and real-world usability rather than purely optimizing metrics in isolation.
