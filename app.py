import streamlit as st
import joblib
import numpy as np

# 1. Load the "brain"
try:
    model = joblib.load("loan_model.pkl")
except:
    st.error("Model file 'loan_model.pkl' not found. Please ensure it is uploaded to your GitHub repository.")

# 2. Set up the "Face" of the App
st.set_page_config(page_title="Alpha Dreamers Banking", page_icon="💰")
st.title("Alpha Dreamers: Loan Default Risk Predictor")
st.markdown("### Decision Support System for Loan Officers")
st.write("Enter the applicant's profile details below to assess potential default risk.")

st.divider()

# 3. Create Input Fields (Organized in two columns)
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    age = st.slider("Age (Years)", 18, 100, 30)
    experience = st.slider("Total Work Experience (Years)", 0, 50, 5)
    job_yrs = st.slider("Years in Current Job", 0, 40, 2)

with col2:
    house_yrs = st.slider("Years in Current House", 0, 20, 5)
    car = st.selectbox("Owns a Car?", ["no", "yes"])
    married = st.selectbox("Marital Status", ["single", "married"])

st.divider()

# 4. Prediction Logic
if st.button("Calculate Risk Assessment"):
    # Convert text inputs to numbers (Encoding)
    car_val = 1 if car == "yes" else 0
    married_val = 1 if married == "married" else 0
    
    # Create the feature array 
    # Format: [Income, Age, Experience, Job_Yrs, House_Yrs, Car, Married]
    features = np.array([[income, age, experience, job_yrs, house_yrs, car_val, married_val]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # 5. Show Results
    if prediction[0] == 1:
        st.error("🚨 **RESULT: HIGH RISK**")
        st.write("This applicant has a high probability of defaulting on the loan.")
    else:
        st.success("✅ **RESULT: LOW RISK**")
        st.write("This applicant is classified as a safe borrower.")

st.info("Note: This tool is an AI assistant. Final decisions should be reviewed by a human loan officer.")
