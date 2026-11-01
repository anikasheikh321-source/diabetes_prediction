import streamlit as st
import numpy as np
import joblib
import os

# Page setup
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's information below to predict the likelihood of diabetes.")

# Load pre-trained model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler files: {e}")
    st.stop()

# Layout inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Make prediction
if st.button("Predict Diabetes Risk", type="primary"):
    # Assemble inputs into matching array format
    raw_inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scaler transformation
    scaled_inputs = scaler.transform(raw_inputs)
    
    # Model prediction & probability
    prediction = model.predict(scaled_inputs)[0]
    probability = model.predict_proba(scaled_inputs)[0][1]
    
    # Visual separator
    st.markdown("---")
    
    if prediction == 1:
        st.error(f"🚨 **High Risk of Diabetes** (Confidence: {probability:.2%})")
        st.warning("The model suggests the patient may be diabetic. Please consult a healthcare professional.")
    else:
        st.success(f"✨ **Low Risk of Diabetes** (Confidence: {1 - probability:.2%})")
        st.info("The model suggests the patient is unlikely to be diabetic.")
