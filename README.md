# Loan-Default-Analysis
Project 1: Alpha Dreamers Banking - Loan Default Analysis
Student Name: Bonolo Ramolapong
Date Completed: March 7, 2026

### Project Description
This project started with analyzing 252,000 loan records to understand why customers default on personal loans. I created 15 visualizations and a Logistic Regression model to identify high-risk patterns for the Alpha Dreamers Banking Consortium.

Now, in this deployment phase, I turned the model into an **interactive web app** using Streamlit. Bank staff can enter applicant details and get instant default risk predictions — low risk or high risk + probability.

### Key Takeaways from Analysis (from original project)
- Stability matters most: Renters default ~13–15% vs homeowners ~5%.  
- No car and single status are red flags.  
- Risk drops sharply after 8–10 years of job experience or older age.  

### Web App Features
- Inputs: Age, Income, Work Experience, Current Job Years, Current House Years, Car Ownership, Marital Status, House Ownership  
- Output: "Low Risk" or "High Risk of Default" + probability %  
- Uses saved logistic regression model and StandardScaler  
- Built with Python, scikit-learn, joblib, and Streamlit

### Files in this repo
- `app.py` → Streamlit web app code  
- `loan_default_model.pkl` → Trained logistic regression model  
- `scaler.pkl` → StandardScaler for input features  
- `requirements.txt` → Required packages

### How to run locally
1. Clone the repo  
2. Install packages: `pip install -r requirements.txt`  
3. Run: `streamlit run app.py`

### Model Evaluation Summary
Accuracy: ~88%  
Focused on **recall** for the default class (catching risky applicants is more important for the bank than avoiding false positives).

Built as part of Python Data Science module – Deployment task.
