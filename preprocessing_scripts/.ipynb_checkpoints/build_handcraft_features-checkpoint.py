import pandas as pd
import click
import os

@click.command()
@click.option("--input_path", help="Merged data about user (.csv file)", type=click.Path(exists=True))
@click.option("--output_path", help="CSV file to write crafted features to", type=click.Path())
def build_handcraft_features(input_path, output_path):
    df = pd.read_csv(input_path)
    epsilon = 0.001

    # List of required columns
    required_columns = [
        "payment_method_id_count", "payment_plan_days_mean", "actual_amount_paid_mean", 
        "actual_amount_paid_sum", "is_auto_renew_mean", "is_cancel_mean", "is_cancel_sum", 
        "membership_expire_day_of_week_mean", "membership_expire_month_mean", 
        "plan_list_price_mean", "bd", "city"
    ]

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing from the input data: {missing_columns}")

    # Creating a new DataFrame with the hand-crafted features
    new_df = pd.DataFrame({
        "msno": df["msno"],
        "fe_payment_days_div_amount_paid": df["payment_plan_days_mean"] / (df["actual_amount_paid_mean"] + epsilon),
        "fe_mean_autorenew_mul_mean_cancel": df["is_auto_renew_mean"] * df["is_cancel_mean"],
        "fe_amount_div_n_trans": df["actual_amount_paid_mean"] / (df["payment_method_id_count"] + epsilon),
        "fe_total_amount_paid_div_trans_done": df["actual_amount_paid_sum"] / (df["payment_method_id_count"] + epsilon),
        "fe_total_payment_days_div_amount_paid": df["payment_plan_days_mean"] * df["payment_method_id_count"] / (df["actual_amount_paid_sum"] + epsilon),
        "fe_mean_expire_date_mul_plan_price": df["membership_expire_day_of_week_mean"] * df["plan_list_price_mean"],
        "fe_mean_amount_paid_mul_autorenew": df["actual_amount_paid_mean"] * df["is_auto_renew_mean"],
        "fe_total_trans_cancelled_div_trans_done": df["is_cancel_sum"] / (df["payment_method_id_count"] + epsilon),
        "fe_total_trans_cancelled_div_payment_days": df["is_cancel_sum"] / (df["payment_plan_days_mean"] * df["payment_method_id_count"] + epsilon),
        "fe_total_trans_cancelled_mul_amount_paid": df["is_cancel_sum"] * df["actual_amount_paid_sum"],
        "fe_plan_list_price_div_amount_paid": df["plan_list_price_mean"] / (df["actual_amount_paid_mean"] + epsilon),
        "fe_mean_expire_date_div_bd": df["membership_expire_day_of_week_mean"] / (df["bd"] + epsilon),
        "fe_city_mul_amount_paid": df["city"] * df["actual_amount_paid_mean"],
        "fe_city_div_payment_days": df["city"] / (df["payment_plan_days_mean"] + epsilon),
    })

    # Writing the new DataFrame to the output path
    new_df.to_csv(output_path, index=False, float_format='%.6f', chunksize=2000)

if __name__ == "__main__":
    build_handcraft_features()
