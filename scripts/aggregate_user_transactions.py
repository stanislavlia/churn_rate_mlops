import pandas as pd
import click
import os
from datetime import datetime

@click.command()
@click.option("--input_path", default="../raw_data/transactions_v2_sample.csv")
@click.option("--output_dir")
def aggregate_user_transactions(input_path, output_dir):

    if os.path.exists(os.path.join(output_dir, "aggregated_transactions.csv")):
        print("The artifact already exists!")
        return

    user_transactions_df = pd.read_csv(input_path)

    # Ensure columns are treated as strings for date parsing
    user_transactions_df['transaction_date'] = user_transactions_df['transaction_date'].astype(str)
    user_transactions_df['membership_expire_date'] = user_transactions_df['membership_expire_date'].astype(str)

    # Fill missing values with 0 or other appropriate value before aggregation
    user_transactions_df.fillna({
        'payment_method_id': 0,
        'payment_plan_days': 0,
        'plan_list_price': 0,
        'actual_amount_paid': 0,
        'is_auto_renew': 0,
        'is_cancel': 0
    }, inplace=True)

    # Convert date columns to datetime objects
    user_transactions_df['transaction_date'] = pd.to_datetime(user_transactions_df['transaction_date'], format="%Y%m%d", errors='coerce')
    user_transactions_df['membership_expire_date'] = pd.to_datetime(user_transactions_df['membership_expire_date'], format="%Y%m%d", errors='coerce')

    # Features for 'transaction_date'
    user_transactions_df['transaction_day_of_week'] = user_transactions_df['transaction_date'].dt.weekday
    user_transactions_df['transaction_day_of_week_name'] = user_transactions_df['transaction_date'].dt.strftime("%A")
    user_transactions_df['transaction_month'] = user_transactions_df['transaction_date'].dt.month
    user_transactions_df['transaction_month_name'] = user_transactions_df['transaction_date'].dt.strftime("%B")
    user_transactions_df['transaction_is_weekend'] = user_transactions_df['transaction_date'].dt.weekday >= 5
    user_transactions_df['transaction_day_of_month'] = user_transactions_df['transaction_date'].dt.day
    user_transactions_df['transaction_quarter'] = user_transactions_df['transaction_date'].dt.quarter
    user_transactions_df['transaction_day_of_year'] = user_transactions_df['transaction_date'].dt.dayofyear

    # Features for 'membership_expire_date'
    user_transactions_df['membership_expire_day_of_week'] = user_transactions_df['membership_expire_date'].dt.weekday
    user_transactions_df['membership_expire_day_of_week_name'] = user_transactions_df['membership_expire_date'].dt.strftime("%A")
    user_transactions_df['membership_expire_month'] = user_transactions_df['membership_expire_date'].dt.month
    user_transactions_df['membership_expire_month_name'] = user_transactions_df['membership_expire_date'].dt.strftime("%B")
    user_transactions_df['membership_expire_is_weekend'] = user_transactions_df['membership_expire_date'].dt.weekday >= 5
    user_transactions_df['membership_expire_day_of_month'] = user_transactions_df['membership_expire_date'].dt.day
    user_transactions_df['membership_expire_quarter'] = user_transactions_df['membership_expire_date'].dt.quarter
    user_transactions_df['membership_expire_day_of_year'] = user_transactions_df['membership_expire_date'].dt.dayofyear

    # Aggregating the data
    aggregated_df = user_transactions_df.groupby('msno').agg({
        'payment_method_id': 'count',
        'payment_plan_days': ['mean', 'sum'],
        'plan_list_price': 'mean',
        'actual_amount_paid': ['mean', 'sum'],
        'is_auto_renew': 'mean',
        'is_cancel': ['mean', 'sum'],
        'transaction_day_of_week': 'mean',
        'transaction_month': 'mean',
        'transaction_is_weekend': ['mean', 'sum'],
        'transaction_day_of_month': ['mean', 'std'],
        'transaction_day_of_year': ['min', 'max'],
        'membership_expire_day_of_week': 'mean',
        'membership_expire_month': ['mean', 'count']
    })

    # Flatten the column multi-index
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]

    # Reset the index
    aggregated_df.reset_index(inplace=True)

    aggregated_df.to_csv(os.path.join(output_dir, "aggregated_transactions.csv"), index=False, float_format='%.2f')


if __name__ == '__main__':
    aggregate_user_transactions()
