import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.copy()

    # Drop public_reference (ID not useful for ML)
    if 'public_reference' in df.columns:
        df = df.drop('public_reference', axis=1)

    # Drop crossover_car_and_van (not needed anymore)
    if 'crossover_car_and_van' in df.columns:
        df = df.drop('crossover_car_and_van', axis=1)

    # Encoding categorical columns
    cat_columns = ['standard_colour', 'standard_make', 'standard_model',
                   'vehicle_condition', 'body_type', 'fuel_type']

    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Stratify based on 'vehicle_condition' for balanced splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=X['vehicle_condition']
    )

    return X_train, X_test, y_train, y_test, label_encoders

if __name__ == "__main__":
    df = load_data('data/AutoTrader.csv')
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)
    print("Preprocessing complete! Train and test sets are ready.")
