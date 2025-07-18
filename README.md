🏦 Loan Approval Prediction Project
This project predicts whether a loan will be approved or not based on applicant details using Machine Learning.

📌 Overview
This is a classification project using Logistic Regression to predict loan approval status.
The model is trained on historical loan applicant data provided in a CSV file.

📁 Dataset
File: loan_prediction.csv
Rows: 614
Features include: Gender, Marital Status, Income, Education, Credit History, Property Area, etc.
⚙️ Technologies Used
Tool/Library	Purpose
Python	Programming language
Pandas	Data manipulation & analysis
NumPy	Numerical operations
scikit-learn	Machine Learning model
Joblib	Saving the model
Seaborn & Matplotlib	Data visualization
VS Code	Code editor
📊 Exploratory Data Analysis (EDA)
Key graphs include:

Loan approval status
<img width="1919" height="1018" alt="Loan Approval Status" src="https://github.com/user-attachments/assets/54532204-3c6e-418f-92dc-a96ba7e25a06" />

Applicant income distribution 
<img width="1919" height="1008" alt="Applicant Income" src="https://github.com/user-attachments/assets/ff6d7dfa-1692-4654-a760-5f9235d417d4" />

Loan amount vs Education
<img width="1917" height="1011" alt="Loan amount vs Education" src="https://github.com/user-attachments/assets/c3cdd35d-7046-4b96-a208-7bbff01f8d3f" />

Property area vs loan status
<img width="1918" height="1017" alt="Property area vs Loan Status" src="https://github.com/user-attachments/assets/35650bcf-543b-45be-9ab6-d07b9ecc3737" />

Correlation heatmap
<img width="1919" height="1018" alt="Correaltion  Heatmap" src="https://github.com/user-attachments/assets/1edc12d8-b82a-4159-adef-811b8d6b0ff4" />


🧠 Model Details
Algorithm Used: Logistic Regression
Accuracy: ~78.8%
Model File: loan_approval_model.pkl
🚀 Installation & Usage
git clone https://github.com/your-username/loan-approval-prediction.git

cd loan-approval-prediction

pip install -r requirements.txt

python loan_approval.py
