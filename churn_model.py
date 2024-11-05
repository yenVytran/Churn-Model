import streamlit as st
import pandas as pd
import joblib

# Tải scaler và mô hình đã huấn luyện
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Tiêu đề
st.title('Customer Churn Prediction')

# Thông tin đầu vào
st.header("Enter Customer Information")

name = st.text_input("Customer name (enter full name)")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=None)
age = st.number_input("Age", min_value=0, max_value=100, value=None)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=None)
balance = st.number_input("Balance", min_value=0.0, value=None)
num_of_products = st.number_input("Number of Products", min_value=0, value=None)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=None)
geography = st.text_input("Geography (enter country name)")
has_cr_card = st.selectbox("Has Credit Card?", options=[None,"Yes", "No"],format_func=lambda x: "Select" if x is None else x)
is_active_member = st.selectbox("Is Active Member?", options=[None,"Yes", "No"],format_func=lambda x: "Select" if x is None else x)
gender = st.selectbox("Gender", options=[None,"Female", "Male"],format_func=lambda x: "Select" if x is None else x)

# Dự đoán
if st.button("Predict"):
    data = {}
    if any(v is None for v in [credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, geography, gender]):
        st.error("Please fill in all fields before predicting.")
    else:
    # Tạo dự liệu đầu vào
        data = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": 1 if geography.lower() == "germany" else 0,
            "Geography_Spain": 1 if geography.lower() == "spain" else 0,
            "Gender_Male": 1 if gender == "Male" else 0
        }

    # Biến đổi dữ liệu đầu vào thành dataframe
    input_df = pd.DataFrame([data])

    # Chuẩn hóa dữ liệu
    X = scaler.transform(input_df)

    # Thực hiện dự đoán
    prediction = model.predict(X)[0]
    st.subheader("Prediction Result")
    result_message = f"{name} is predicted to be {'EXITED' if prediction == 1 else 'NOT EXITED'}."
    st.write(result_message)
