# 📱 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)

## 🎯 Business Objective
Customer churn is a critical metric for telecom companies. This project builds a Machine Learning pipeline to predict which customers are likely to cancel their service. 

**Impact:** By identifying at-risk customers, the marketing team can trigger retention campaigns (discounts, personalized offers), potentially saving the company thousands in lost revenue.

## 📊 Data Overview
The project uses the **Telco Customer Churn** dataset. 
- **Target:** `Churn` (Yes/No)
- **Features:** Tenure, Monthly Charges, Contract Type, Internet Service, etc.

## 🛠️ Tech Stack
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Random Forest Classifier
- **Model Deployment:** Joblib for serialization

## 📈 Key Findings
- **Contract Type:** Month-to-month customers are the most likely to churn.
- **Tenure:** Newer customers (0-6 months) have a significantly higher churn rate.
- **Feature Importance:** `TotalCharges` and `Tenure` are the strongest predictors.

## 🚀 How to Use
1. Clone this repo: `git clone https://github.com/yourusername/Customer-Churn-Prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python main.py`