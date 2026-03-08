import streamlit as st
import joblib
import numpy as np

# Load the model
# Ensure 'loan_model.pkl' is uploaded to your GitHub in the same folder
model = joblib.load("loan_model.pkl")

st.title("Alpha Dreamers: Loan Risk Predictor")

# 1. Inputs
income = st.number_input("Annual Income", min_value=0, value=50000)
age = st.slider("Age", 18, 100, 30)
experience = st.slider("Work Experience (Total Years)", 0, 50, 5)
job_yrs = st.slider("Years in Current Job", 0, 40, 2)
house_yrs = st.slider("Years in Current House", 0, 20, 5)
car = st.selectbox("Owns a Car?", ["no", "yes"])
married = st.selectbox("Marital Status", ["single", "married"])

if st.button("Predict Risk"):
    car_val = 1 if car == "yes" else 0
    married_val = 1 if married == "married" else 0

    features = np.array([[income, age, experience, job_yrs, house_yrs, car_val, married_val]])

    # Get the Risk Probability
    risk_score = model.predict_proba(features)[0][1]

    st.write(f"### Internal Risk Score: {risk_score:.2f}")

    # FINAL THRESHOLD: 0.50
    # This lets the 40yr old (0.49) pass and catches the 18yr old (>0.50)
    if risk_score > 0.50:
        st.error(f"Result: High Risk (Score: {risk_score:.2f})")
    else:
        st.success(f"Result: Low Risk (Score: {risk_score:.2f})")
