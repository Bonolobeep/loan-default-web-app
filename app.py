
import streamlit as st
import joblib
import numpy as np

# Load the saved full model and scaler (the original 403-feature version)
model = joblib.load('loan_default_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Default Risk Predictor")
st.subheader("Alpha Dreamers Banking Consortium")
st.write("Enter the applicant's details below to check the risk of default.")

# ── User inputs (only the easy ones we can collect) ──
age = st.slider("Age (years)", 18, 80, 35)
income = st.number_input("Annual Income", 10000, 2000000, 500000)
experience = st.slider("Total Work Experience (years)", 0, 40, 5)
current_job_yrs = st.slider("Years in Current Job", 0, 30, 5)
current_house_yrs = st.slider("Years in Current House", 0, 20, 5)

car_own = st.radio("Owns a Car?", ["yes", "no"])
marital = st.radio("Marital Status", ["single", "married"])
house_own = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])

if st.button("Predict Risk"):
    # ── Prepare values ──
    car_val = 1 if car_own == "yes" else 0
    marital_val = 1 if marital == "married" else 0
    house_rented = 1 if house_own == "rented" else 0
    house_owned = 1 if house_own == "owned" else 0
    house_norent_noown = 1 if house_own == "norent_noown" else 0

    # ── Create full 403-column input array filled with zeros ──
    final_input = np.zeros((1, 403))

    # ── Fill the known columns (positions from your feature_names_in_) ──
    # Position 0: Income
    # Position 1: Age
    # Position 2: Experience
    # Position 3: Married/Single
    # Position 4: Car_Ownership
    # Position 5: CURRENT_JOB_YRS
    # Position 6: CURRENT_HOUSE_YRS
    # Position 7: House_Ownership_owned
    # Position 8: House_Ownership_rented
    # (Position 9 would be House_Ownership_norent_noown if present — adjust if needed)

    final_input[0, 0] = income
    final_input[0, 1] = age
    final_input[0, 2] = experience
    final_input[0, 3] = marital_val
    final_input[0, 4] = car_val
    final_input[0, 5] = current_job_yrs
    final_input[0, 6] = current_house_yrs
    final_input[0, 7] = house_owned
    final_input[0, 8] = house_rented

    # ── Scale only the numeric columns (positions 0,1,2,5,6) ──
    numeric_indices = [0, 1, 2, 5, 6]
    numeric_values = final_input[0, numeric_indices].reshape(1, -1)
    scaled_numeric = scaler.transform(numeric_values)
    final_input[0, numeric_indices] = scaled_numeric[0]

    # ── All other columns (professions, cities, states) stay 0 — that's OK for demo ──

    # ── Make prediction ──
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]  # probability of default (class 1)

    if pred == 1:
        st.error(f"**High Risk of Default** — Probability: {prob:.1%}")
        st.write("Recommendation: Consider additional checks or higher interest.")
    else:
        st.success(f"**Low Risk** — Probability of default: {prob:.1%}")
        st.write("Recommendation: Suitable for standard approval.")

    st.markdown("**From analysis**: Renters, singles, and short job/house tenure increase risk significantly.")
