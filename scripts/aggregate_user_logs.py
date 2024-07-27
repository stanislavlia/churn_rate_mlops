import pandas as pd
import numpy as np
import click


@click.command()
@click.option("--input_path", default="../raw_data/user_logs_v2_sample.csv")
@click.option("--output_dir")
def aggregate_user_logs(input_path, output_dir):
    print("HEllo ", input_path)



if __name__ == '__main__':
    aggregate_user_logs()