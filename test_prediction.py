import joblib

# Load the saved model
model = joblib.load("loan_approval_model.pkl")

# Sample new input (you can change values as needed)
# Format: [Gender, Married, Dependents, Education, Self_Employed,
# ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
# Credit_History, Property_Area]

new_applicant = [[1, 1, 1, 0, 0, 5000, 2000, 150, 360, 1.0, 2]]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("✅ Loan Approved!")
else:
    print("❌ Loan Not Approved.")
