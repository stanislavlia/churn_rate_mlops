import polars as pl
import numpy as np
import click


@click.command()
@click.option("--merged_data_path",
              default="../processed_data/users_data_merged.csv")
@click.option("--train_labels_path")
@click.option("--output_dir", default="../processed_data/")
def build_train_data(merged_data_path,
                     train_labels_path,
                     output_dir):

    merged_data_df = pl.scan_csv(merged_data_path)
    train_labels_df = pl.scan_csv(train_labels_path)

    train_df = train_labels_df.join(merged_data_df, on="msno", how="left")

    train_df.sink_csv(output_dir + "full_train_data.csv",
                      batch_size=100,
                      float_precision=32,
                      maintain_order=False,
                      type_coercion=True,
                      simplify_expression=True)


if __name__ == "__main__":
    build_train_data()
