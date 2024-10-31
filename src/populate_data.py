import requests
import pandas as pd
import numpy as np

def fetch_historical_race_data(season):
    """
    Fetch historical race data for a given season from the Ergast API.
    
    Parameters:
    - season: int, the season year to fetch data for.
    
    Returns:
    - pd.DataFrame: DataFrame containing race results.
    """
    url = f"http://ergast.com/api/f1/{season}/results.json"
    response = requests.get(url)
    data = response.json()
    races = data['MRData']['RaceTable']['Races']

    # Create a DataFrame for race results
    race_data = pd.DataFrame(races)
    
    return race_data

def create_mock_telemetry_data(num_records):
    """
    Create mock telemetry data for races.
    
    Parameters:
    - num_records: int, the number of records to generate.
    
    Returns:
    - pd.DataFrame: DataFrame containing mock telemetry data.
    """
    telemetry_data = {
        'lap': np.random.randint(1, 60, num_records),
        'speed': np.random.uniform(80, 220, num_records),
        'brake': np.random.uniform(0, 1, num_records),
        'throttle': np.random.uniform(0, 1, num_records),
        'gear': np.random.randint(1, 8, num_records),
        'race_id': np.random.randint(1, 100, num_records)  # Mock race IDs
    }
    
    return pd.DataFrame(telemetry_data)

def create_mock_weather_data(num_records):
    """
    Create mock weather data for races.
    
    Parameters:
    - num_records: int, the number of records to generate.
    
    Returns:
    - pd.DataFrame: DataFrame containing mock weather data.
    """
    weather_data = {
        'race_id': np.random.randint(1, 100, num_records),
        'temperature': np.random.uniform(10, 35, num_records),  # in Celsius
        'condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy'], num_records),
        'humidity': np.random.uniform(30, 100, num_records)  # in percentage
    }
    
    return pd.DataFrame(weather_data)

def main():
    season = 2021  # Example season
    # Fetch historical race data
    race_data = fetch_historical_race_data(season)
    race_data.to_csv('data/historical_race_data.csv', index=False)

    # Create mock telemetry and weather data
    telemetry_data = create_mock_telemetry_data(len(race_data) * 10)  # 10 telemetry records per race
    telemetry_data.to_csv('data/telemetry_data.csv', index=False)

    weather_data = create_mock_weather_data(len(race_data))  # One weather record per race
    weather_data.to_csv('data/weather_data.csv', index=False)

    print("Data population complete!")

if __name__ == "__main__":
    main()
