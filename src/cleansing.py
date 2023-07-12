import pandas as pd

from src import utils


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.simplify_weather_columns(df)
    df = utils.convert_date_to_datetime(df)
    return df
