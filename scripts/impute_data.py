import polars as pl
import click
import os
import json


def impute_missing_values(df, imputation_config):
    for column, value in imputation_config.items():
        if value is not None:
            df = df.with_columns(pl.col(column).fill_null(value))
    return df


@click.command()
@click.option("--data", help="Path to csv file with merged data")
@click.option("--imput_config", help="Prepared imputation config JSON")
@click.option("--output", help="Output file to write imputed data")
def impute_data(data, imput_config, output):
    
    df = pl.read_csv(data)
    
    with open(imput_config, "r") as f:
        config = dict(json.load(f))
        
    
    df = impute_missing_values(df, config)
    
    df.write_csv(output)
    
if __name__ == "__main__":
    impute_data()
    
    