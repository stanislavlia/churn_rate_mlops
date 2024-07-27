import polars as pl
import click
import os


@click.command()
@click.option("--input_path", default="../raw_data/user_logs_v2_sample.csv")
@click.option("--output_dir")
def aggregate_user_logs(input_path, output_dir):

    if os.path.exists(os.path.join(output_dir, "aggregated_logs.csv")):
        print("The artifact already exists!")
        return

    user_logs_df = pl.scan_csv(input_path)

    # Fill missing values with 0 or other appropriate value before aggregation
    user_logs_df = user_logs_df.with_columns([
        pl.col('num_25').fill_null(0),
        pl.col('num_50').fill_null(0),
        pl.col('num_75').fill_null(0),
        pl.col('num_985').fill_null(0),
        pl.col('num_100').fill_null(0),
        pl.col('num_unq').fill_null(0),
        pl.col('total_secs').fill_null(0),
    ])

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
