import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/RandomForest_model.pkl')  # Or whatever best model you saved

# Load the real dataset
df = pd.read_csv('data/AutoTrader.csv')  # Update path if needed!

# Prepare main lists
makes = sorted(df['standard_make'].unique())
colours = sorted(df['standard_colour'].unique())
vehicle_conditions = sorted(df['vehicle_condition'].unique())
fuel_types = sorted(df['fuel_type'].unique())

# Create dictionary for make → models
make_to_models = {
    make: sorted(df[df['standard_make'] == make]['standard_model'].unique())
    for make in makes
}

# Create dictionary for make → body types
make_to_body_types = {
    make: sorted(df[df['standard_make'] == make]['body_type'].unique())
    for make in makes
}

# Streamlit App
st.title("🚗 Advanced Dynamic Car Price Prediction App")

with st.form("car_form"):
    # Select Make
    selected_make = st.selectbox("Select Car Make", makes)

    # Dynamically filter Models and Body Types based on selected Make
    models_available = make_to_models.get(selected_make, [])
    selected_model = st.selectbox("Select Car Model", models_available)

    body_types_available = make_to_body_types.get(selected_make, [])
    selected_body_type = st.selectbox("Select Body Type", body_types_available)

    mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, step=500)
    selected_colour = st.selectbox("Color", colours)
    selected_vehicle_condition = st.selectbox("Vehicle Condition", vehicle_conditions)
    year_of_registration = st.slider(
        "Year of Registration",
        min_value=int(df['year_of_registration'].min()),
        max_value=int(df['year_of_registration'].max()),
        value=2015
    )
    selected_crossover = st.selectbox("Crossover Car and Van?", ['No', 'Yes'])
    selected_fuel_type = st.selectbox("Fuel Type", fuel_types)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Manual encoding similar to LabelEncoder used during training
    input_data = {
        'mileage': mileage,
        'standard_colour': list(colours).index(selected_colour),
        'standard_make': list(makes).index(selected_make),
        'standard_model': list(models_available).index(selected_model),
        'vehicle_condition': list(vehicle_conditions).index(selected_vehicle_condition),
        'year_of_registration': year_of_registration,
        'body_type': list(body_types_available).index(selected_body_type),
        'crossover_car_and_van': 1 if selected_crossover == 'Yes' else 0,
        'fuel_type': list(fuel_types).index(selected_fuel_type)
    }

    input_df = pd.DataFrame([input_data])

    with st.spinner('Predicting...'):
        prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: £{prediction:,.2f}")
