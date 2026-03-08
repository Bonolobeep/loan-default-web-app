
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("loan_model.pkl")

# Dataset-informed threshold
THRESHOLD = 0.12

st.title("Alpha Dreamers: Loan Risk Predictor")
st.write("Enter client details below to predict loan risk.")

# Input features
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

    try:
        risk_score = model.predict_proba(features)[0][1]
        st.write(f"### Internal Risk Score: {risk_score:.2f}")

        if risk_score >= THRESHOLD:
            st.error("🚨 High Risk Loan Applicant")
        else:
            st.success("✅ Low Risk Loan Applicant")
    except Exception as e:
        st.error("Prediction failed. Please check model.")
        st.write(str(e))
