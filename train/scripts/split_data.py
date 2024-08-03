import os
import click
import pandas as pd

@click.command()
@click.option("--data", help="Path to csv file with full training data", required=True)
@click.option("--output_dir", help="Directory to write splits in", required=True)
@click.option("--train_size", help="Fraction of training data", default=0.6, type=float)
@click.option("--val_size", help="Fraction of validation data", default=0.2, type=float)
def split_data(data, output_dir, train_size, val_size):
    df = pd.read_csv(data)

    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    size = df.shape[0]
    
    df = df.sort_values(by="registration_init_time")

    # Drop 'Unnamed: 0' if it exists
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    
    train_idx_end = int(size * train_size)
    val_idx_end = train_idx_end + int(size * val_size)
    
    train_df = df[:train_idx_end]
    val_df = df[train_idx_end:val_idx_end]
    test_df = df[val_idx_end:]
    
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
if __name__ == "__main__":
    split_data()
