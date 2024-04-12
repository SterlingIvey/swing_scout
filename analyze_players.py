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
    df = df.dropna(subset=['today_rank', 'yesterday_rank', 'last_week_rank', 'score'])
    return df

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

def main():
    df = load_data('player_data.csv')
    df = preprocess_data(df)
    
    features = df[['today_rank', 'yesterday_rank', 'last_week_rank', 'score']]
    target = df['points']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Cross validation
    cv_scores = cross_val_score(model, features, target, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", np.mean(cv_scores))

if __name__ == "__main__":
    main()