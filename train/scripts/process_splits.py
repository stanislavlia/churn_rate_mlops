import click
from sklearn.preprocessing import StandardScaler
import os
import json
import joblib
import pandas as pd

ARTIFACTS_PATH = "/home/stanislav/Desktop/churn_forecast/artifacts"

def build_scaler(df, numeric_features_to_scale, artifacts_dir):
    numeric_df = df[numeric_features_to_scale]
    scaler = StandardScaler()

    scaler.fit(numeric_df)

    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))
    with open(os.path.join(artifacts_dir, "features_to_scale.json"), "w") as f:
        json.dump({"features": numeric_features_to_scale}, f)

def preprocess(df, train=True, artifacts_path=ARTIFACTS_PATH):
    TO_DROP = ["msno", "registration_init_time", "last_transaction_date"]
    CATEGORICAL_FEATURES = ["city", "gender", "registered_via"]

    df["gender"] = df["gender"].replace({"male": 1, "female": -1, "MISSING": 0})
    df = df.drop(TO_DROP, axis=1)

    if df.empty:
        print("Warning: The DataFrame is empty after preprocessing. No scaling applied.")
        return df  # or handle as needed

    if 'is_churn' in df.columns:
        target = df['is_churn'].copy()
        df = df.drop('is_churn', axis=1)
    else:
        target = None

    features_to_scale = [col for col in df.columns if col not in CATEGORICAL_FEATURES]

    if train:
        scaler = StandardScaler()
        scaler.fit(df[features_to_scale])
        joblib.dump(scaler, os.path.join(artifacts_path, "scaler.joblib"))

    scaler = joblib.load(os.path.join(artifacts_path, "scaler.joblib"))

    df[features_to_scale] = scaler.transform(df[features_to_scale])

    if target is not None:
        df['is_churn'] = target

    return df

@click.command()
@click.option("--splits_dir", help="Dir with splits")
def preprocessing(splits_dir):
    train_df = pd.read_csv(os.path.join(splits_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test_split.csv"))

    processed_train_df = preprocess(train_df, train=True, artifacts_path=ARTIFACTS_PATH)
    processed_val_df = preprocess(val_df, train=False, artifacts_path=ARTIFACTS_PATH)
    processed_test_df = preprocess(test_df, train=False, artifacts_path=ARTIFACTS_PATH)

    processed_train_df.to_csv(os.path.join(splits_dir, "train_processed.csv"), index=False)
    processed_val_df.to_csv(os.path.join(splits_dir, "val_processed.csv"), index=False)
    processed_test_df.to_csv(os.path.join(splits_dir, "test_processed.csv"), index=False)

if __name__ == "__main__":
    preprocessing()
