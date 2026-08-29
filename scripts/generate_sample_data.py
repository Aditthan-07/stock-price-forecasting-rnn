"""
Generates synthetic stock market dataset matching Kaggle "Stock Market Data" schema (Date, Open, High, Low, Close, Adj Close, Volume)
spanning 1,000 trading days for reproducible testing.
"""

import os
import numpy as np
import pandas as pd


def generate_stock_data(output_csv_path: str, num_days: int = 1000, start_date: str = '2019-01-01', initial_price: float = 100.0, seed: int = 42):
    np.random.seed(seed)
    date_range = pd.bdate_range(start=start_date, periods=num_days)
    
    # Generate daily returns using Geometric Brownian Motion simulation
    mu = 0.0003     # Daily trend / drift
    sigma = 0.015   # Daily volatility
    
    daily_returns = np.random.normal(mu, sigma, num_days)
    price_series = initial_price * np.exp(np.cumsum(daily_returns))
    
    records = []
    for i in range(num_days):
        close_p = price_series[i]
        daily_range = np.abs(np.random.normal(0, 0.01)) * close_p
        open_p = close_p + np.random.uniform(-0.5, 0.5) * daily_range
        high_p = max(open_p, close_p) + np.random.uniform(0.1, 0.5) * daily_range
        low_p = min(open_p, close_p) - np.random.uniform(0.1, 0.5) * daily_range
        adj_close_p = close_p * (1.0 - np.random.uniform(0, 0.005))
        volume = int(np.random.normal(5000000, 1000000))
        volume = max(1000000, volume)
        
        records.append({
            'Date': date_range[i].strftime('%Y-%m-%d'),
            'Open': round(open_p, 2),
            'High': round(high_p, 2),
            'Low': round(low_p, 2),
            'Close': round(close_p, 2),
            'Adj Close': round(adj_close_p, 2),
            'Volume': volume
        })
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Sample stock data generated with {len(df)} rows at '{output_csv_path}'.")
    return df


if __name__ == '__main__':
    csv_path = 'data/sample_stock_data.csv'
    generate_stock_data(csv_path)
