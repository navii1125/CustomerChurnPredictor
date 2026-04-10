📊 Customer Churn Prediction System
📌 Overview

This project predicts whether a customer is likely to churn based on service usage and billing features using Machine Learning.

🚀 Features
Data preprocessing and feature encoding
Logistic Regression model for prediction
Model evaluation using Accuracy and ROC-AUC
Interactive UI using Streamlit
Analytics dashboard with charts and trends
Explainable AI using feature importance

🛠️ Tech Stack
Python
Pandas
Scikit-learn
Streamlit
Matplotlib
Joblib

📊 Input Features
Tenure
Monthly Charges
Total Charges
Contract Type
Internet Service
Payment Method

📈 Output
Churn Prediction (Yes/No)
Churn Probability Score

▶️ How to Run
pip install -r requirements.txt
python src/train.py
streamlit run src/app.py
