import polars as pl
import click
import os
import pandas as pd

def compute_timing_features(df):
    df["registration_init_time"] = pd.to_datetime(df["registration_init_time"], format="%Y%m%d")
    df["last_transaction_date"] = pd.to_datetime(df["last_transaction_date"], format="%Y-%m-%d")
    df['registration_year'] = df['registration_init_time'].dt.year
    df['registration_month'] = df['registration_init_time'].dt.month
    df['last_transaction_year'] = df['last_transaction_date'].dt.year
    df['last_transaction_month'] = df['last_transaction_date'].dt.month
    df['usage_period_days'] = (df["last_transaction_date"] - df["registration_init_time"]).dt.days
    
    return df
    

@click.command()
@click.option("--merged_data_path", help="Path to the merged data file")
@click.option("--train_labels_path", help="Path to the training labels file")
@click.option("--output_dir", default="processed_data/", help="Directory to save the output data")
def build_train_data(merged_data_path, train_labels_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    merged_df = pd.read_csv(merged_data_path)
    train_labels_df = pd.read_csv(train_labels_path)

    train_df = merged_df.merge(train_labels_df, on="msno", how="left")

    train_df = compute_timing_features(train_df)
    
    output_path = os.path.join(output_dir, "full_train_data.csv")
    train_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    build_train_data()