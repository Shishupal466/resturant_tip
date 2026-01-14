import streamlit as st
import pandas as pd
import pickle

st.color_picker = "#3C1C1C"
# Load Model
model = pickle.load(open('model.pkl', 'rb'))

st.title("💰 Restaurant Tip Prediction App")

st.write("Predict tip amount based on customer and order details.")

# User Inputs
total_bill = st.number_input('Total Bill Amount (₹)', min_value=300.0, max_value=6000.0)
size = st.slider('Group Size', 1, 6, 1)
sex = st.selectbox('Gender', ['Male', 'Female'])
smoker = st.selectbox('Smoker', ['Yes', 'No'])
day = st.selectbox('Day', ['Thur', 'Fri', 'Sat', 'Sun'])
time = st.selectbox('Time', ['Lunch', 'Dinner'])

# Convert to model‑readable format
input_df = pd.DataFrame({
    'total_bill': [total_bill],
    'sex': [sex],
    'smoker': [smoker],
    'day': [day],
    'time': [time],
    'size': [size]
})

# Handle categorical values (IMPORTANT: must match training encoding)
input_df = pd.get_dummies(input_df)

# Align with model columns
# (only needed if training had more dummy columns)
model_columns = model.feature_names_in_
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict Tip"):
    prediction = model.predict(input_df)[0] * 10
    st.success(f"Estimated Tip: 💵 **₹{prediction:.2f}**")
    st.info("Note: Accuracy of the model is around 80%. Predictions may vary.")
    st.text("Due to limited data, predictions might not be precise.")
