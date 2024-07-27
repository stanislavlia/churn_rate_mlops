import polars as pl
import numpy as np
import click
import os


@click.command()
@click.option("--members_path", help="Path of members.csv file")
@click.option("--agg_logs_path",
              help="Path to aggregated logs of users that we will merge with")
@click.option("--agg_transactions_path",
              help="Path to aggregated transactions")
@click.option("--output_dir", help="Directory with data artifacts")
def merge_users_data(members_path,
                     agg_logs_path,
                     agg_transactions_path,
                     output_dir):

    if os.path.exists(os.path.join(output_dir, "users_data_merged.csv")):
        print("The artifact already exists!")
        return

    members_df = pl.scan_csv(members_path)
    agg_logs_df = pl.scan_csv(agg_logs_path)
    agg_transactions_df = pl.scan_csv(agg_transactions_path)

    members_logs_df = members_df.join(agg_logs_df, on="msno", how="left")
    members_logs_transactions = members_logs_df.join(
        agg_transactions_df, on="msno", how="left")

    # stream merged table to csv file
    members_logs_transactions.sink_csv(output_dir + "users_data_merged.csv",
                                       batch_size=100,
                                       float_precision=32,
                                       maintain_order=False,
                                       type_coercion=True,
                                       simplify_expression=True)


if __name__ == '__main__':
    merge_users_data()
