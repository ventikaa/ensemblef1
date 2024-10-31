from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.hyperparameter_tuning import tune_hyperparameters
from src.model_evaluation import evaluate_model

def main():
    historical_data, telemetry_data, weather_data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(historical_data, telemetry_data, weather_data)
    
    X_train = create_features(X_train)
    X_test = create_features(X_test)

    rf_model, xgb_model = train_models(X_train, y_train)
    
    # Hyperparameter tuning example
    rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    best_rf_model = tune_hyperparameters(rf_model, rf_param_grid, X_train, y_train)
    
    evaluate_model(best_rf_model, X_test, y_test)

if __name__ == "__main__":
    main()

