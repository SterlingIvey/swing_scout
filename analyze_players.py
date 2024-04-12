import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Pull today's file. Static now but will be dynamic eventually
def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Largest changes in player rankings from day to day and week to week
    df['day_change'] = abs(df['today_rank'] - df['yesterday_rank'])
    df['week_change'] = abs(df['today_rank'] - df['last_week_rank'])
    df = df.dropna(subset=['today_rank', 'yesterday_rank', 'last_week_rank', 'score']), inplace=True
    return df

def largest_change_players(df):
    # Find the 3 of your players with the largest value changes
    max_day_change = df[df['day_change'] == df['day_change'].max()]
    max_week_change = df[df['week_change'] == df['week_change'].max()]
    return max_day_change, max_week_change
