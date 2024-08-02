import pandas as pd
import click
import os
from datetime import datetime
import numpy as np
from scipy import stats



@click.command()
@click.option("--input_path", default="../raw_data/user_logs_v2_sample.csv")
@click.option("--output_dir")
def aggregate_user_logs(input_path, output_dir):

    if os.path.exists(os.path.join(output_dir, "aggregated_logs.csv")):
        print("The artifact already exists!")
        return

    user_logs_df = pd.read_csv(input_path)

    # Fill missing values with 0 or other appropriate value before aggregation
    user_logs_df.fillna({
        'num_25': 0,
        'num_50': 0,
        'num_75': 0,
        'num_985': 0,
        'num_100': 0,
        'num_unq': 0,
        'total_secs': 0
    }, inplace=True)

    # Convert date column to datetime objects
    user_logs_df['date'] = pd.to_datetime(user_logs_df['date'], format="%Y%m%d", errors='coerce')

    # Extract additional features from the date
    user_logs_df['day_of_week'] = user_logs_df['date'].dt.weekday
    user_logs_df['day_of_week_name'] = user_logs_df['date'].dt.strftime("%A")
    user_logs_df['month'] = user_logs_df['date'].dt.month
    user_logs_df['month_name'] = user_logs_df['date'].dt.strftime("%B")
    user_logs_df['is_weekend'] = user_logs_df['date'].dt.weekday >= 5
    user_logs_df['day_of_month'] = user_logs_df['date'].dt.day
    user_logs_df['quarter'] = user_logs_df['date'].dt.quarter
    user_logs_df['day_of_year'] = user_logs_df['date'].dt.dayofyear

    # Aggregate the data
    aggregated_df = user_logs_df.groupby('msno').agg({
        'num_25': 'mean',
        'num_50': ['mean', 'min', 'max', 'std'],
        'num_75': 'mean',
        'num_985': 'mean',
        'num_100': 'mean',
        'num_unq': 'mean',
        'total_secs': ['mean', 'std'],
        'month': "mean",
        'is_weekend': ['mean', 'count', 'sum'],
        'quarter': "mean",
        'day_of_week': "mean",
        'day_of_year': ['min', 'max', "mean", 'count']
    })

    # Flatten the column multi-index
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]

    # Reset the index
    aggregated_df.reset_index(inplace=True)

    # Save the aggregated dataframe
    aggregated_df.to_csv(os.path.join(output_dir, "aggregated_logs.csv"), index=False, float_format='%.2f')


if __name__ == '__main__':
    aggregate_user_logs()
