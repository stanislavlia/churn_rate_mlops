import polars as pl
import numpy as np
import click


@click.command()
@click.option("--merged_data_path", default="../processed_data/users_data_merged.csv")
@click.option("--train_labels_path")
@click.option("--output_dir", default="../processed_data/")
def build_train_data(merged_data_path,
                     train_labels_path,
                     output_dir):
    pass