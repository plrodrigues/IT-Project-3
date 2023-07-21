import pandas as pd


def get_the_time_diff_of_the_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    date_time_column = kwargs.get("date_time_column", "date_time")

    df_sorted = df.sort_values(by=date_time_column)

    return df_sorted[[date_time_column]].diff()


def is_the_sampling_frequency_constant(df: pd.DataFrame, **kwargs) -> bool:
    date_time_column = kwargs.get("date_time_column", "date_time")

    df_diff = get_the_time_diff_of_the_dataframe(df, **kwargs)

    is_equal_to_zero = df_diff[date_time_column].std().total_seconds() == 0

    if is_equal_to_zero:
        return True
    else:
        return False


def get_the_sampling_frequency_hz_and_period_sec(df: pd.DataFrame, **kwargs) -> tuple[float, float]:
    date_time_column = kwargs.get("date_time_column", "date_time")

    df_diff = get_the_time_diff_of_the_dataframe(df, **kwargs)

    period_sec = df_diff[date_time_column].dt.total_seconds().mean()

    sampling_freq_hz = 1 / period_sec

    return sampling_freq_hz, period_sec
