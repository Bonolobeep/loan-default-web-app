import streamlit as st
import joblib
import numpy as np

# 1. Load the model
model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="Loan Risk Predictor", page_icon="💰")
st.title("Alpha Dreamers: Loan Default Risk Predictor")

# 2. Input form
income = st.number_input("Annual Income", min_value=0, value=50000)
age = st.slider("Age", 18, 100, 30)
experience = st.slider("Work Experience (Years)", 0, 50, 5)
job_yrs = st.slider("Years in Current Job", 0, 40, 2)
house_yrs = st.slider("Years in Current House", 0, 20, 5)

car = st.selectbox("Owns a Car?", ["no", "yes"])
married = st.selectbox("Marital Status", ["single", "married"])

# 3. Prediction Button
if st.button("Predict Risk"):
    car_val = 1 if car == "yes" else 0
    married_val = 1 if married == "married" else 0

    features = np.array([[income, age, experience, job_yrs, house_yrs, car_val, married_val]])

    # Get the probability (how likely is a default?)
    # predict_proba returns [chance of safe, chance of risky]
    probability = model.predict_proba(features)[0][1] 

    st.write(f"**Risk Probability:** {probability:.2f}")

    # Use a 0.5 threshold (standard)
    if probability > 0.5:
        st.error(f"Result: High Risk (Score: {probability:.2f})")
    else:
        st.success(f"Result: Low Risk (Score: {probability:.2f})")
