import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from preprocess_data import load_data, preprocess_data


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models_params = {
        'RandomForest': (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': [30, 50]
            }
        ),
        'XGBoost': (
            XGBRegressor(random_state=42, verbosity=0),
            {
                'n_estimators': [30, 50],
                'max_depth': [6, 10]
            }
        ),
        'DecisionTree': (
            DecisionTreeRegressor(random_state=42),
            {

                'min_samples_split': [2, 5]
            }
        )
    }

    results = {}

    for name, (model, param_grid) in models_params.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'BestParams': grid.best_params_,
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2
        }

        # Save each best model
        joblib.dump(best_model, f'models/{name}_model.pkl', compress=('zlib', 3))
        print(f"✅ {name} model saved.")

    # Ensemble Model
    print("\nTraining Ensemble Model...")

    rf_best = joblib.load('models/RandomForest_model.pkl')
    xgb_best = joblib.load('models/XGBoost_model.pkl')
    dt_best = joblib.load('models/DecisionTree_model.pkl')

    ensemble = VotingRegressor(estimators=[
        ('rf', rf_best),
        ('xgb', xgb_best),
        ('dt', dt_best)
    ])

    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_ensemble)
    rmse = mean_squared_error(y_test, y_pred_ensemble) ** 0.5
    r2 = r2_score(y_test, y_pred_ensemble)

    results['Ensemble'] = {
        'BestParams': 'Voting of best models',
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2
    }

    joblib.dump(ensemble, 'models/Ensemble_model.pkl', compress=('zlib', 3))
    print("✅ Ensemble model saved.")

    results_df = pd.DataFrame(results).T
    return results_df

if __name__ == "__main__":
    df = load_data('data/AutoTrader.csv')
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)
