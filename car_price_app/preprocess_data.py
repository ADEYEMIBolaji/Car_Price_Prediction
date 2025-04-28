import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

def load_data(path):
    df = pd.read_csv(path, dtype=str)  # Force all columns as string
    return df

def preprocess_data(df):
    df = df.copy()

    # Drop unnamed extra columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Convert selected numeric columns
    numeric_columns = ['mileage', 'year_of_registration', 'price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop public_reference if it exists
    if 'public_reference' in df.columns:
        df = df.drop('public_reference', axis=1)

    # Drop crossover_car_and_van if it exists
    if 'crossover_car_and_van' in df.columns:
        df = df.drop('crossover_car_and_van', axis=1)

    # List of categorical columns
    cat_columns = ['standard_colour', 'standard_make', 'standard_model',
                   'vehicle_condition', 'body_type', 'fuel_type']

    # Transform categorical columns to UPPERCASE
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    # Label Encode
    label_encoders = {}
    for col in cat_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['vehicle_condition']
    )

    return X_train, X_test, y_train, y_test, label_encoders

if __name__ == "__main__":
    df = load_data('data/AutoTrader.csv')
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)
    print("✅ Preprocessing complete! Train and test sets ready.")
