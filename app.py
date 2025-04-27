import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
# Load the trained model
model = joblib.load('models/XGBoost_model.pkl')  # Or whatever best model you saved

# Load the real dataset
df = pd.read_csv('data/AutoTrader.csv')  # Update path if needed!

# Prepare main dynamic lists
makes = sorted(df['standard_make'].unique())
colours = sorted(df['standard_colour'].unique())
vehicle_conditions = [condition for condition in sorted(df['vehicle_condition'].unique()) if condition in ['New', 'Used']]
fuel_types = sorted(df['fuel_type'].unique())

# Prepare make → models
make_to_models = {
    make: sorted(df[df['standard_make'] == make]['standard_model'].unique())
    for make in makes
}

# Prepare make → body types
make_to_body_types = {
    make: sorted(df[df['standard_make'] == make]['body_type'].unique())
    for make in makes
}

# Get current year
current_year = datetime.now().year

# Streamlit App
st.title("🚗 Dynamic Car Price Prediction App")

# Select Make
selected_make = st.selectbox("Select Car Make", makes)

# Based on selected Make
models_available = make_to_models.get(selected_make, [])
selected_model = st.selectbox("Select Car Model", models_available)

body_types_available = make_to_body_types.get(selected_make, [])
selected_body_type = st.selectbox("Select Body Type", body_types_available)

# Organize inputs into 2 columns
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, step=500)
    selected_colour = st.selectbox("Color", colours)
    selected_vehicle_condition = st.selectbox("Vehicle Condition", vehicle_conditions)

with col2:
    year_of_registration = st.slider(
        "Year of Registration",
        min_value=2000,
        max_value=current_year,
        value=2015
    )
    selected_fuel_type = st.selectbox("Fuel Type", fuel_types)

# Predict button
if st.button("Predict Price"):
    # Manual encoding (assuming same as LabelEncoder order)
    input_data = {
        'mileage': mileage,
        'standard_colour': list(colours).index(selected_colour),
        'standard_make': list(makes).index(selected_make),
        'standard_model': list(models_available).index(selected_model),
        'vehicle_condition': list(vehicle_conditions).index(selected_vehicle_condition),
        'year_of_registration': year_of_registration,
        'body_type': list(body_types_available).index(selected_body_type),
        'fuel_type': list(fuel_types).index(selected_fuel_type)
    }

    input_df = pd.DataFrame([input_data])

    with st.spinner('Predicting...'):
        prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: £{prediction:,.2f}")
