import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('car_sales_model.pkl')

# App title
st.title("Car Sales Prediction App")

# App description
st.write("Provide the car features to predict the expected sales.")

# Input fields for user to provide feature values
price = st.number_input('Price (in thousands)', min_value=0.0, max_value=100.0, step=0.1)
engine_size = st.number_input('Engine Size', min_value=0.0, max_value=10.0, step=0.1)
horsepower = st.number_input('Horsepower', min_value=0.0, max_value=1000.0, step=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency (MPG)', min_value=0.0, max_value=100.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    # Input array for the model
    features = np.array([[price, engine_size, horsepower, fuel_efficiency]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display the prediction
    st.success(f'Predicted Sales: {prediction[0]:.2f} units')

# To run the app: streamlit run app.py
