import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    historical_data = pd.read_csv('data/historical_race_data.csv')
    telemetry_data = pd.read_csv('data/telemetry_data.csv')
    weather_data = pd.read_csv('data/weather_data.csv')
    
    return historical_data, telemetry_data, weather_data

def preprocess_data(historical_data, telemetry_data, weather_data):
    # Merge datasets on relevant keys
    merged_data = historical_data.merge(telemetry_data, on='race_id').merge(weather_data, on='race_id')

    # Handle missing values, encoding, etc.
    merged_data.fillna(method='ffill', inplace=True)

    # Define features and target variable
    X = merged_data.drop(columns=['race_outcome'])  # Replace with actual target column
    y = merged_data['race_outcome']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


