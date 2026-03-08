import streamlit as st
import joblib
import numpy as np

# 1. Load the simplified model you just saved
model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="Loan Risk Predictor", page_icon="💰")
st.title("Alpha Dreamers: Loan Default Risk Predictor")
st.write("Enter the details below to see if an applicant is a high-risk or low-risk borrower.")

# 2. Create the input form
# We use the exact 7 features from your 'simple_features' list
income = st.number_input("Annual Income", min_value=0, value=50000)
age = st.slider("Age", 18, 100, 30)
experience = st.slider("Work Experience (Years)", 0, 50, 5)
job_yrs = st.slider("Years in Current Job", 0, 40, 2)
house_yrs = st.slider("Years in Current House", 0, 20, 5)

car = st.selectbox("Owns a Car?", ["no", "yes"])
married = st.selectbox("Marital Status", ["single", "married"])

# 3. Prediction Button
if st.button("Predict Risk"):
    # Convert text to numbers (Encoding)
    car_val = 1 if car == "yes" else 0
    married_val = 1 if married == "married" else 0
    
    # Create the array for the model
    # MUST be in the same order: Income, Age, Experience, Job_Yrs, House_Yrs, Car, Married
    features = np.array([[income, age, experience, job_yrs, house_yrs, car_val, married_val]])
    
    # Get the prediction
    prediction = model.predict(features)
    
    # Show result
    if prediction[0] == 1:
        st.error("Result: High Risk (Likely to Default)")
    else:
        st.success("Result: Low Risk (Safe Borrower)")
