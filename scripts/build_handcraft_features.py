import click
import polars as pl
import os


@click.command()
@click.option("--input_path", help="Merged data about user (.csv file)",
              type=click.Path(exists=True))
@click.option("--output_path",
              help="CSV file to write crafted features to", type=click.Path())
def build_handcraft_features(input_path, output_path):
    df = pl.read_csv(input_path)
    epsilon = 1e-5

    new_df = pl.DataFrame({
        "msno": df["msno"],
        "fe_payment_days_div_amount_paid": df["mean_payment_plan_days"] / (df["mean_actual_amount_paid"] + epsilon),
        "fe_mean_autorenew_mul_mean_cancel": df["mean_is_auto_renew"] * df["mean_is_cancel"],
        "fe_amount_div_n_trans": df["mean_actual_amount_paid"] / (df["number_transactions_done"] + epsilon),
        "fe_total_amount_paid_div_trans_done": df["total_actual_amount_paid"] / (df["number_transactions_done"] + epsilon),
        "fe_total_payment_days_div_amount_paid": df["total_payment_plan_days"] / (df["total_actual_amount_paid"] + epsilon),
        "fe_mean_expire_date_mul_plan_price": df["mean_membership_expire_date"] * df["mean_plan_list_price"],
        "fe_mean_amount_paid_mul_autorenew": df["mean_actual_amount_paid"] * df["mean_is_auto_renew"],
        "fe_total_trans_cancelled_div_trans_done": df["total_transactions_cancelled"] / (df["number_transactions_done"] + epsilon),
        "fe_total_trans_cancelled_div_payment_days": df["total_transactions_cancelled"] / (df["total_payment_plan_days"] + epsilon),
        "fe_total_trans_cancelled_mul_amount_paid": df["total_transactions_cancelled"] * df["total_actual_amount_paid"],
        "fe_plan_list_price_div_amount_paid": df["mean_plan_list_price"] / (df["mean_actual_amount_paid"] + epsilon),
        "fe_mean_expire_date_div_bd": df["mean_membership_expire_date"] / (df["bd"] + epsilon),
        "fe_city_mul_amount_paid": df["city"] * df["mean_actual_amount_paid"],
        "fe_city_div_payment_days": df["city"] / (df["mean_payment_plan_days"] + epsilon),
    })

    new_df.write_csv(output_path)


if __name__ == "__main__":
    build_handcraft_features()
