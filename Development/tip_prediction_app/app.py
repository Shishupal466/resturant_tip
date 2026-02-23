# import streamlit as st
# import pandas as pd
# import pickle

# st.color_picker = "#3C1C1C"
# # Load Model
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("💰 Restaurant Tip Prediction App")

# st.write("Predict tip amount based on customer and order details.")

# # User Inputs
# total_bill = st.number_input('Total Bill Amount (₹)', min_value=300.0, max_value=6000.0)
# size = st.slider('Group Size', 1, 6, 1)
# sex = st.selectbox('Gender', ['Male', 'Female'])
# smoker = st.selectbox('Smoker', ['Yes', 'No'])
# day = st.selectbox('Day', ['Thur', 'Fri', 'Sat', 'Sun'])
# time = st.selectbox('Time', ['Lunch', 'Dinner'])

# # Convert to model‑readable format
# input_df = pd.DataFrame({
#     'total_bill': [total_bill],
#     'sex': [sex],
#     'smoker': [smoker],
#     'day': [day],
#     'time': [time],
#     'size': [size]
# })

# # Handle categorical values (IMPORTANT: must match training encoding)
# input_df = pd.get_dummies(input_df)

# # Align with model columns
# # (only needed if training had more dummy columns)
# model_columns = model.feature_names_in_
# input_df = input_df.reindex(columns=model_columns, fill_value=0)

# # Prediction
# if st.button("Predict Tip"):
#     prediction = model.predict(input_df)[0] * 10
#     st.success(f"Estimated Tip: 💵 **₹{prediction:.2f}**")
#     st.info("Note: Accuracy of the model is around 80%. Predictions may vary.")
#     st.text("Due to limited data, predictions might not be precise.")









import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Restaurant Tip Predictor",
    page_icon="💰",
    layout="centered"
)

# Custom CSS for professional UI
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton > button {
    background-color: #2E8B57;
    color: white;
    font-size: 16px;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #246b45;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Title
st.markdown("<h1 style='text-align:center;'>💰 Restaurant Tip Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart AI predicts tip based on customer details</p>", unsafe_allow_html=True)

st.write("")

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    total_bill = st.number_input('💵 Total Bill Amount (₹)', min_value=300.0, max_value=6000.0)
    size = st.slider('👥 Group Size', 1, 6, 1)
    sex = st.selectbox('🧑 Gender', ['Male', 'Female'])

with col2:
    smoker = st.selectbox('🚬 Smoker', ['Yes', 'No'])
    day = st.selectbox('📅 Day', ['Thur', 'Fri', 'Sat', 'Sun'])
    time = st.selectbox('🕒 Time', ['Lunch', 'Dinner'])

# Input DataFrame
input_df = pd.DataFrame({
    'total_bill': [total_bill],
    'sex': [sex],
    'smoker': [smoker],
    'day': [day],
    'time': [time],
    'size': [size]
})

# Encode categorical
input_df = pd.get_dummies(input_df)

# Align columns with model
if hasattr(model, "feature_names_in_"):
    model_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict Button
st.write("")
if st.button("🎯 Predict Tip"):

    prediction = model.predict(input_df)[0] * 10

    # Result Card
    st.markdown(f"""
    <div class="card">
        <h2 style="color:#2E8B57;">💵 Estimated Tip: ₹{prediction:.2f}</h2>
        <p>Model accuracy ~80% | Predictions may vary</p>
    </div>
    """, unsafe_allow_html=True)

    # Chart Visualization
    st.write("")
    st.subheader("📊 Tip vs Total Bill Comparison")

    chart_df = pd.DataFrame({
        "Category": ["Total Bill", "Predicted Tip"],
        "Amount (₹)": [total_bill, prediction]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_df["Category"], chart_df["Amount (₹)"])
    ax.set_ylabel("Amount in ₹")
    ax.set_title("Bill vs Tip Visualization")

    st.pyplot(fig)
