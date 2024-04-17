import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """
    Load player data from a CSV file.
    
    Parameters:
    filepath (str): The path to the CSV file.
    
    Returns:
    DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully from {}".format(filepath))
        return df
    except FileNotFoundError:
        logging.error("The file at {} does not exist.".format(filepath))
        return None

def preprocess_data(df):
    """
    Preprocess the data by calculating day-to-day and week-to-week changes.
    
    Parameters:
    df (DataFrame): The DataFrame containing player ranks.
    
    Returns:
    DataFrame: Preprocessed DataFrame.
    """
    if df is not None:
        df['day_change'] = abs(df['today_rank'] - df['yesterday_rank'])
        df['week_change'] = abs(df['today_rank'] - df['last_week_rank'])
        df = df.dropna(subset=['today_rank', 'yesterday_rank', 'last_week_rank', 'score'])
        logging.info("Data preprocessed")
        return df
    else:
        logging.error("Input DataFrame is None")
        return None

def largest_change_players(df):
    """
    Identify players with the largest changes in rankings.
    
    Parameters:
    df (DataFrame): The DataFrame to analyze.
    
    Returns:
    tuple: DataFrames of players with the largest day-to-day and week-to-week changes.
    """
    if df is not None:
        max_day_change = df[df['day_change'] == df['day_change'].max()]
        max_week_change = df[df['week_change'] == df['week_change'].max()]
        return max_day_change, max_week_change
    else:
        logging.error("Input DataFrame is None")
        return None, None
    
def main():
    df = load_data('player_data.csv')
    if df is not None:
        df = preprocess_data(df)
        max_day_change, max_week_change = largest_change_players(df)
        # Further processing or saving results can be done here

if __name__ == "__main__":
    main()