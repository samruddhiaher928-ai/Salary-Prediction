import streamlit as st
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("salary_model.pkl")
encoder = joblib.load("education_encoder.pkl")

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Title
st.title("💼 Employee Salary Prediction")

st.write("Enter employee details below to predict the estimated salary.")

# User Inputs
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=2.0,
    step=0.5
)

education = st.selectbox(
    "Education",
    encoder.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=25
)

# Predict Button
if st.button("Predict Salary"):

    education_encoded = encoder.transform([education])[0]

    input_data = pd.DataFrame({
        "YearsExperience": [experience],
        "Education": [education_encoded],
        "Age": [age]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ₹ {prediction:,.0f} per year")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")