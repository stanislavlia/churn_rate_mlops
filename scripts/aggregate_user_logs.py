import pandas as pd
import polars as pl
import numpy as np
import click


@click.command()
@click.option("--input_path", default="../raw_data/user_logs_v2_sample.csv")
@click.option("--output_dir")
def aggregate_user_logs(input_path, output_dir):
    
    user_logs_df = pl.scan_csv(input_path)
    


    aggregated_df = user_logs_df.group_by('msno').agg([
    pl.col('date').mean().alias('mean_date'),
    pl.col('date').count().alias('count_of_logs'),
    pl.col('num_25').mean().alias('mean_num_25'),
    pl.col('num_50').mean().alias('mean_num_50'),
    pl.col('num_75').mean().alias('mean_num_75'),
    pl.col('num_985').mean().alias('mean_num_985'),
    pl.col('num_100').mean().alias('mean_num_100'),
    pl.col('num_unq').mean().alias('mean_num_unq'),
    pl.col('total_secs').mean().alias('mean_total_secs')
    ])


    aggregated_df.sink_csv(output_dir + "aggregated_logs.csv",
                           batch_size=100,
                           float_precision=32,
                           maintain_order=False,
                           type_coercion=True,
                           simplify_expression=True)
    



if __name__ == '__main__':
    aggregate_user_logs()
