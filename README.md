# Loan Default Risk Web App

**Student**: Bonolo Ramolapong  
**Date**: March 2026  
**Project**: Python Data Science – Model Deployment Task

### Overview
Interactive Streamlit web app that predicts personal loan default risk for Alpha Dreamers Banking Consortium.  
Uses logistic regression trained on ~252,000 customer records.

### Key Insights from Analysis
- Stability is critical: Renters default ~13–15% vs homeowners ~5%  
- Singles, no car, short job tenure (<8–10 years) = higher risk  
- Risk decreases with age and longer experience

### Web App Features
- Inputs: Age, Income, Experience, Job Years, House Years, Car Ownership, Marital Status, House Ownership  
- Output: Low/High risk + default probability (%)  
- Uses pre-trained model and StandardScaler

### Technologies
- Python, scikit-learn, joblib, pandas, numpy  
- Streamlit for the web interface

### Files
- `app.py` – Streamlit app code  
- `loan_default_model.pkl` – Trained logistic regression model  
- `scaler.pkl` – Feature scaler  
- `requirements.txt` – Dependencies

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
