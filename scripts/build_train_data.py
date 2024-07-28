import polars as pl
import click
import os

@click.command()
@click.option("--merged_data_path", help="Path to the merged data file")
@click.option("--train_labels_path", help="Path to the training labels file")
@click.option("--output_dir", default="processed_data/", help="Directory to save the output data")
def build_train_data(merged_data_path, train_labels_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    merged_df = pl.read_csv(merged_data_path)
    train_labels_df = pl.read_csv(train_labels_path)

    train_df = merged_df.join(train_labels_df, on="msno", how="left")
    
    output_path = os.path.join(output_dir, "full_train_data.csv")
    train_df.write_csv(output_path)

if __name__ == "__main__":
    build_train_data()
