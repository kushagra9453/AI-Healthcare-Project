import streamlit as st
import pickle
import os

st.set_page_config(page_title="AI Healthcare", layout="wide")

st.title("🧠 AI Powered Healthcare Prediction System")

st.markdown("""
### 📌 Project Overview
This system predicts diseases using Machine Learning:
- Diabetes
- Heart Disease
- Parkinson’s
""")

# Safe model loader
def load_model(path):
    try:
        return pickle.load(open(path, 'rb'))
    except:
        return None

working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = load_model(f'{working_dir}/saved_models/diabetes_model.sav')
heart_model = load_model(f'{working_dir}/saved_models/heart_disease_model.sav')
parkinsons_model = load_model(f'{working_dir}/saved_models/parkinsons_model.sav')

option = st.sidebar.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinsons"])

# -------- DIABETES --------
if option == "Diabetes":
    st.header("Diabetes Prediction")

    val = st.text_input("Enter any number")

    if st.button("Predict"):
        if diabetes_model:
            try:
                data = [float(val)] * 8
                res = diabetes_model.predict([data])
                st.success("Diabetic" if res[0] == 1 else "Not Diabetic")
            except:
                st.warning("Enter valid number")
        else:
            st.error("Model not loaded")

# -------- HEART --------
if option == "Heart Disease":
    st.header("Heart Disease Prediction")

    val = st.text_input("Enter any number")

    if st.button("Predict"):
        if heart_model:
            try:
                data = [float(val)] * 13
                res = heart_model.predict([data])
                st.success("Heart Disease" if res[0] == 1 else "No Disease")
            except:
                st.warning("Enter valid number")
        else:
            st.error("Model not loaded")

# -------- PARKINSONS --------
if option == "Parkinsons":
    st.header("Parkinsons Prediction")

    val = st.text_input("Enter any number")

    if st.button("Predict"):
        if parkinsons_model:
            try:
                data = [float(val)] * 22
                res = parkinsons_model.predict([data])
                st.success("Parkinsons" if res[0] == 1 else "No Parkinsons")
            except:
                st.warning("Enter valid number")
        else:
            st.error("Model not loaded")