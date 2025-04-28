import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
# Load the trained model

model = joblib.load('models/Ensemble_model.pkl')  # Or whatever best model you saved


# Load the real dataset
df = pd.read_csv('data/AutoTrader.csv', dtype=str)  # Update path if needed!

# Prepare dynamic lists (convert all to UPPERCASE just once)
makes = sorted(df['standard_make'].astype(str).str.upper().unique())
colours = sorted(df['standard_colour'].astype(str).str.upper().unique())
vehicle_conditions = [condition for condition in sorted(df['vehicle_condition'].astype(str).str.upper().unique()) if condition in ['NEW', 'USED']]
fuel_types = sorted(df['fuel_type'].astype(str).str.upper().unique())

# Prepare make → models mapping (also UPPERCASE)
make_to_models = {
    make: sorted(df[df['standard_make'].str.upper() == make]['standard_model'].astype(str).str.upper().unique())
    for make in makes
}

# Prepare make → body types mapping
make_to_body_types = {
    make: sorted(df[df['standard_make'].str.upper() == make]['body_type'].astype(str).str.upper().unique())
    for make in makes
}

# Get current year
current_year = datetime.now().year

# Streamlit App
st.title("🚗 Findout Car Estimate")

# Select Make
selected_make = st.selectbox("Select Car Make", makes)

# Dynamically filter models and body types based on selected make
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
        value=current_year
    )
    selected_fuel_type = st.selectbox("Fuel Type", fuel_types)

# Predict button
if st.button("Predict Price"):
    # Force all inputs to UPPERCASE before encoding
    selected_make = selected_make.upper()
    selected_model = selected_model.upper()
    selected_body_type = selected_body_type.upper()
    selected_colour = selected_colour.upper()
    selected_vehicle_condition = selected_vehicle_condition.upper()
    selected_fuel_type = selected_fuel_type.upper()

    # Encoding input manually
    input_data = {
        'mileage': mileage,
        'standard_colour': colours.index(selected_colour),
        'standard_make': makes.index(selected_make),
        'standard_model': models_available.index(selected_model),
        'vehicle_condition': vehicle_conditions.index(selected_vehicle_condition),
        'year_of_registration': year_of_registration,
        'body_type': body_types_available.index(selected_body_type),
        'fuel_type': fuel_types.index(selected_fuel_type)
    }

    input_df = pd.DataFrame([input_data])

    with st.spinner('Predicting...'):
        prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: £{prediction:,.2f}")
