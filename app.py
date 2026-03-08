import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")

st.title("Alpha Dreamers: Loan Risk Predictor")

# Input form
income = st.number_input("Annual Income", min_value=0, value=50000)
age = st.slider("Age", 18, 100, 30)
experience = st.slider("Work Experience (Years)", 0, 50, 5)
job_yrs = st.slider("Years in Current Job", 0, 40, 2)
house_yrs = st.slider("Years in Current House", 0, 20, 5)

car = st.selectbox("Owns a Car?", ["no", "yes"])
married = st.selectbox("Marital Status", ["single", "married"])

if st.button("Predict Risk"):
    car_val = 1 if car == "yes" else 0
    married_val = 1 if married == "married" else 0
    features = np.array([[income, age, experience, job_yrs, house_yrs, car_val, married_val]])

    # Get the probability of risk
    probability = model.predict_proba(features)[0][1] 

    st.write(f"**Calculated Risk Score:** {probability:.2f}")

    #  SAFETY THRESHOLD: 0.25
   
    if probability > 0.25:
        st.error(f"Result: High Risk (Score: {probability:.2f})")
        st.write("⚠️ This applicant has a low stability score.")
    else:
        st.success(f"Result: Low Risk (Score: {probability:.2f})")
