import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# App Title and Header
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.markdown("""
Welcome to the **AI Loan Eligibility Checker** 💡  
Fill in the details below to predict whether your loan will be approved or not.
""")

# Sidebar for basic info
st.sidebar.header("📘 About")
st.sidebar.markdown("""
This app uses a **Machine Learning model (Logistic Regression)**  
to predict if a loan application is **Approved or Not Approved**  
based on applicant details.
""")

# Input fields for user data
st.subheader("🧾 Applicant Details")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["No", "Yes"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=100)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=100)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, step=1)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0, step=10)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Convert categorical inputs to numeric (same encoding as training)
data = {
    'Gender': 1 if Gender == 'Male' else 0,
    'Married': 1 if Married == 'Yes' else 0,
    'Dependents': 3 if Dependents == '3+' else int(Dependents),
    'Education': 0 if Education == 'Graduate' else 1,
    'Self_Employed': 1 if Self_Employed == 'Yes' else 0,
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': 0 if Property_Area == 'Rural' else (1 if Property_Area == 'Semiurban' else 2)
}

# Create DataFrame for prediction
new_applicant = pd.DataFrame([data])

# Prediction button
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(new_applicant)
    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Not Approved.")
        st.snow()

# Footer
st.markdown("---")
st.markdown("Created by **Meenakshi Palai**")
