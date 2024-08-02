import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import click

SPLITS_PATH = "/"

@click.command()
@click.option("--model_type", help='Supported models [xgboost, catboost, lightgbm, rf]')
@click.option("--run_name")
#...
def tune_params(model_type, run_name, n_trials, time_limit):
    pass