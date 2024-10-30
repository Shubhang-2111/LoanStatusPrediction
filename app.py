import streamlit as st

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.title("Predict Loan Status")

model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('pickle/person_gender.pkl', 'rb') as file:
    person_gender_encoder = pickle.load(file)

with open('pickle/prev_loan.pkl', 'rb') as file:
    prev_loan_encoder = pickle.load(file)

with open('pickle/one_hot_encoded.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('pickle/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


person_age = st.number_input("Person Age", min_value=18, max_value=120, value=30)
person_gender = st.selectbox("Gender",["male","female"])
person_income = st.number_input("Person Income", min_value=0, value=50000)
person_emp_exp = st.number_input("Person Employment Experience (in years)", min_value=0, value=5)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0, value=20.0)
cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, value=10)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
previous_loan_defaults_on_file = st.selectbox("Previous Loan",["Yes","No"])


# Input fields for education (categorical)
education_options = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
education_selection = st.selectbox("Education Level", education_options)

# Input fields for home ownership (categorical)
home_ownership_options = ["RENT", "OWN", "MORTGAGE", "OTHER"]
home_ownership_selection = st.selectbox("Home Ownership Type", home_ownership_options)

# Input fields for loan intent (categorical)
loan_intent_options = [
    "DEBTCONSOLIDATION", 
    "EDUCATION", 
    "HOMEIMPROVEMENT", 
    "MEDICAL", 
    "PERSONAL", 
    "VENTURE"
]


loan_intent_selection = st.selectbox("Loan Intent", loan_intent_options)

data = pd.DataFrame({
    'person_age':[person_age],
    'person_gender':[person_gender],
    'person_education':[education_selection],
    'person_income':[person_income],
    'person_emp_exp':[person_emp_exp],
    'person_home_ownership':[home_ownership_selection],
    'loan_amnt':[loan_amnt],
    'loan_intent':[loan_intent_selection],
    'loan_int_rate':[loan_int_rate],
    'loan_percent_income':[loan_percent_income],
    'cb_person_cred_hist_length':[cb_person_cred_hist_length],
    'credit_score':[credit_score],
    'previous_loan_defaults_on_file':[previous_loan_defaults_on_file]
})


data['person_gender'] = person_gender_encoder.transform(data['person_gender'])
data['previous_loan_defaults_on_file'] = prev_loan_encoder.transform(data['previous_loan_defaults_on_file'])


one_hot_encoded = one_hot_encoder.transform(data[['person_education','person_home_ownership','loan_intent']]).toarray()

one_hot_encoded_df = pd.DataFrame(one_hot_encoded,columns=one_hot_encoder.get_feature_names_out())

data = pd.concat([data.drop(['person_education','person_home_ownership','loan_intent'],axis = 1),one_hot_encoded_df],axis=1)

scaled_data = scaler.transform(data)

probability = model.predict(scaled_data)


if st.button("Predict Loan status"):
    st.write(probability)
    if probability > 0.5:
        st.write("Loan Sanctioned")
    else:
        st.write("Loan not sanctioned")