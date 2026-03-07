
import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('loan_default_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Default Risk Predictor")
st.subheader("Alpha Dreamers Banking Consortium")
st.write("Enter the applicant's details below to check the risk of default.")

# Simple inputs matching your key features
age = st.slider("Age (years)", 18, 80, 35)
income = st.number_input("Annual Income", 10000, 2000000, 500000)
experience = st.slider("Total Work Experience (years)", 0, 40, 5)
current_job_yrs = st.slider("Years in Current Job", 0, 30, 5)
current_house_yrs = st.slider("Years in Current House", 0, 20, 5)

car_own = st.radio("Owns a Car?", ["yes", "no"])
marital = st.radio("Marital Status", ["single", "married"])
house_own = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])

if st.button("Predict Risk"):
    # Prepare input in the same order as your training data
    car_val = 1 if car_own == "yes" else 0
    marital_val = 1 if marital == "married" else 0
    house_rented = 1 if house_own == "rented" else 0
    house_owned = 1 if house_own == "owned" else 0

    # Numeric part first (for scaling)
    raw_numeric = np.array([[income, age, experience, current_job_yrs, current_house_yrs]])
    scaled_numeric = scaler.transform(raw_numeric)

    # Add the binary/one-hot parts
    binary_part = np.array([[car_val, marital_val, house_rented, house_owned]])

    # Combine
    final_input = np.hstack([scaled_numeric, binary_part])

    # Make prediction
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]  # probability of default (class 1)

    if pred == 1:
        st.error(f"**High Risk of Default** — Probability: {prob:.1%}")
        st.write("Recommendation: Consider additional checks or higher interest.")
    else:
        st.success(f"**Low Risk** — Probability of default: {prob:.1%}")
        st.write("Recommendation: Suitable for standard approval.")

    st.markdown("**From analysis**: Renters, singles, and short job/house tenure increase risk significantly.")
