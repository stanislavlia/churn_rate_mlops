import os
import click
import pandas as pd

@click.command()
@click.option("--data", help="Path to csv file with full training data")
@click.option("--output_dir", help="Directory to write splits in")
@click.option("--train_size", help="Fraction of training data", default=0.9)
def split_data(data, output_dir, train_size):

    df = pd.read_csv(data)
 
    size = df.shape[0]
    
    train_idx_end = int(size * train_size)
    test_idx = int( (size - train_idx_end) / 2)
    
    #sort values by date of user registration
    df = df.sort_values(by="registration_init_time")
    try:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    except:
        print("Unnamed not found")
    
    train_df = df[:train_idx_end]
    
    val_df = df[train_idx_end: test_idx]
    test_df = df[test_idx:]
    
    #save splits to csv
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
   
if __name__ == "__main__":
    split_data()


    