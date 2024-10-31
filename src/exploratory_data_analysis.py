import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file.
    
    Returns:
    - pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def plot_histogram(data, column, bins=30):
    """
    Plot histogram for a specified column.
    
    Parameters:
    - data: pd.DataFrame, the input dataset.
    - column: str, the column name to plot.
    - bins: int, number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_correlation_matrix(data):
    """
    Plot a heatmap of the correlation matrix.
    
    Parameters:
    - data: pd.DataFrame, the input dataset.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

def summary_statistics(data):
    """
    Print summary statistics of the dataset.
    
    Parameters:
    - data: pd.DataFrame, the input dataset.
    """
    print("Summary Statistics:")
    print(data.describe())

def plot_boxplot(data, column, by=None):
    """
    Plot boxplot for a specified column, optionally grouped by another column.
    
    Parameters:
    - data: pd.DataFrame, the input dataset.
    - column: str, the column name to plot.
    - by: str, optional grouping column.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=by, y=column, data=data)
    plt.title(f'Boxplot of {column}' + (f' by {by}' if by else ''))
    plt.xlabel(by)
    plt.ylabel(column)
    plt.show()

def main():
    # Load dataset
    data = load_data('data/historical_race_data.csv')  # Update with your file path

    # Perform EDA
    summary_statistics(data)

    # Plot histograms for numeric features
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        plot_histogram(data, col)

    # Plot correlation matrix
    plot_correlation_matrix(data)

    # Boxplot for specific features (adjust as necessary)
    if 'race_position' in data.columns and 'weather_condition' in data.columns:
        plot_boxplot(data, 'race_position', by='weather_condition')

if __name__ == "__main__":
    main()

