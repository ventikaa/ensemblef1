import pandas as pd

def create_features(X):
    """
    Function to create new features based on the existing data.
    
    Parameters:
    - X: pd.DataFrame, the input feature set.
    
    Returns:
    - X: pd.DataFrame, the modified feature set with new features.
    """

    # 1. Average lap time (assuming lap times are recorded for each lap)
    if 'lap_time' in X.columns:
        X['avg_lap_time'] = X['lap_time'].mean()
        X['median_lap_time'] = X['lap_time'].median()
        X['std_lap_time'] = X['lap_time'].std()

    # 2. Create a binary feature for rain condition
    if 'weather_condition' in X.columns:
        X['is_rainy'] = (X['weather_condition'] == 'Rain').astype(int)  # 1 if rainy, else 0
        X['weather_condition_encoded'] = X['weather_condition'].astype('category').cat.codes  # Encode categorical

    # 3. Calculate the difference between grid position and race position
    if 'grid_position' in X.columns and 'race_position' in X.columns:
        X['position_change'] = X['grid_position'] - X['race_position']
        X['final_position'] = X['race_position'].apply(lambda x: 1 if x == 1 else 0)  # 1 if finished 1st

    # 4. Track temperature impact (if temperature data is available)
    if 'temperature' in X.columns:
        X['temp_effect'] = X['temperature'].apply(lambda x: 1 if x > 25 else 0)  # Example: 1 if temp > 25°C
        X['temp_mean'] = X['temperature'].mean()
        X['temp_max'] = X['temperature'].max()
        X['temp_min'] = X['temperature'].min()

    # 5. Create time-related features (if timestamp data is available)
    if 'race_timestamp' in X.columns:
        X['race_year'] = pd.to_datetime(X['race_timestamp']).dt.year
        X['race_month'] = pd.to_datetime(X['race_timestamp']).dt.month
        X['race_day'] = pd.to_datetime(X['race_timestamp']).dt.day
        X['race_weekday'] = pd.to_datetime(X['race_timestamp']).dt.weekday

    # 6. Create a feature for the total number of laps completed
    if 'total_laps' in X.columns:
        X['total_laps_completed'] = X['total_laps'].cumsum()  # Cumulative laps over the races

    # 7. Tire-related features (if tire data is available)
    if 'tire_type' in X.columns:
        X['tire_type_encoded'] = X['tire_type'].astype('category').cat.codes  # Encode tire types
        X['tire_age'] = X['lap_number'] - X['tire_change_lap']  # Age of the tire in laps
        
        # Assuming you have data about tire performance
        X['tire_performance'] = X['tire_type_encoded'] * (X['lap_time'] / X['avg_lap_time'])

    # 8. Speed-related features
    if 'speed' in X.columns:
        X['avg_speed'] = X['speed'].mean()
        X['max_speed'] = X['speed'].max()
        X['min_speed'] = X['speed'].min()

    # 9. Driver and Team-related features (if applicable)
    if 'driver_id' in X.columns:
        X['driver_wins'] = X.groupby('driver_id')['final_position'].transform('sum')  # Total wins for each driver
        X['driver_avg_position'] = X.groupby('driver_id')['race_position'].transform('mean')  # Average finishing position

    if 'team_id' in X.columns:
        X['team_avg_position'] = X.groupby('team_id')['race_position'].transform('mean')  # Average position for team
        X['team_wins'] = X.groupby('team_id')['final_position'].transform('sum')  # Total wins for each team

    # 10. Normalization/Standardization of features (if necessary)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Normalize selected features
    cols_to_scale = ['avg_lap_time', 'position_change', 'temperature', 'avg_speed']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    return X
