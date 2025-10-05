import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model.pkl')

st.title("Amazon Delivery Time Predictor")

st.markdown("""
Enter order and agent details below to predict the estimated delivery time in hours.
""")

agent_age = st.number_input("Agent Age", min_value=18, max_value=100, value=30)
agent_rating = st.number_input("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
distance = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)

weather = st.selectbox("Weather (encoded)", options=[0,1,2,3], index=0)
traffic = st.selectbox("Traffic (encoded)", options=[0,1,2], index=0)
vehicle = st.selectbox("Vehicle (encoded)", options=[0,1], index=0)
area = st.selectbox("Area (encoded)", options=[0,1], index=0)
category = st.selectbox("Category (encoded)", options=[0,1,2,3], index=0)

order_hour = st.number_input("Order Hour (0-23)", min_value=0, max_value=23, value=12)
order_day = st.number_input("Order Day of Week (0=Monday)", min_value=0, max_value=6, value=3)

# Prepare input dataframe for model prediction
input_df = pd.DataFrame(
    [[agent_age, agent_rating, distance, weather, traffic, vehicle, area, category, order_hour, order_day]],
    columns=['Agent_Age', 'Agent_Rating', 'Distance', 'Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'OrderHour', 'OrderDayOfWeek']
)

if st.button("Predict Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Delivery Time: {prediction:.2f} hours")
