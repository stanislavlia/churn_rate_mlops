import polars as pl
import numpy as np
import click
import os

@click.command()
@click.option("--input_path", default="../raw_data/transactions_v2.csv")
@click.option("--output_dir")
def aggregate_user_transactions(input_path, output_dir):
    
    user_transactions_df = pl.scan_csv(input_path)
    
    aggregated_df = user_transactions_df.group_by('msno').agg([
    pl.col('payment_method_id').mean().alias('number_transactions_done'),
    pl.col('payment_plan_days').mean().alias('mean_payment_plan_days'),
    pl.col('payment_plan_days').sum().alias('total_payment_plan_days'),
    pl.col('plan_list_price').mean().alias('mean_plan_list_price'),
    pl.col('actual_amount_paid').mean().alias('mean_actual_amount_paid'),
    pl.col('actual_amount_paid').sum().alias('total_actual_amount_paid'),
    pl.col('is_auto_renew').mean().alias('mean_is_auto_renew'),
    pl.col('transaction_date').mean().alias('mean_transaction_date'),
    pl.col('membership_expire_date').mean().alias('mean_membership_expire_date'),
    pl.col('is_cancel').mean().alias('mean_is_cancel'),
    pl.col('is_cancel').sum().alias('total_transactions_cancelled'),
])


    aggregated_df.sink_csv(output_dir + "aggregated_transactons.csv",
                           batch_size=100,
                           float_precision=32,
                           maintain_order=False,
                           type_coercion=True,
                           simplify_expression=True)
    



if __name__ == '__main__':
    aggregate_user_transactions()
