import polars as pl
import click
import os
import json

zero_fill_features=["number_transactions_done",
                    "total_transactions_cancelled", 
                    "mean_is_cancel"]

@click.command()
@click.option("--train_data", help="Csv file with all merged train data")
@click.option("--config_output", help="JSON to write imputation config")
def compute_imputation_dict(train_data, config_output):
    """
    Computes an imputation dictionary for a given DataFrame.

    Args:
    - df (pl.DataFrame): The DataFrame to compute imputation strategies for.
    - zero_fill_features (list of str): List of features to fill with zero. 
                                         If None, no features will be filled with zero.

    Returns:
    - dict: A dictionary with column names as keys and imputation strategies as values.
    """
    df = pl.read_csv(train_data)
    

    imputation_dict = {}

    for column in df.columns:
        if column in zero_fill_features:
            imputation_dict[column] = 0
        elif column == "gender":
            imputation_dict[column] = "MISSING"
        else:
            imputation_dict[column] = df[column].mean()

    with open(config_output, "w") as f:
        json.dump(imputation_dict, f, indent=3)

if __name__ == "__main__":
    

    compute_imputation_dict()
    
