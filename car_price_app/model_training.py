import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from preprocess_data import load_data, preprocess_data


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'LinearRegression': LinearRegression()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({
            'model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2
        })

        # Save each model
        joblib.dump(model, f'models/{name}_model.pkl',compress=3)
        print(f"{name} model saved.")

    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    df = load_data('data/AutoTrader.csv')
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)
