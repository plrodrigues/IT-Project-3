import numpy as np
import pandas as pd

NEW_COLUMN_NAMES = {
    "Weather": "weather",
    "Date/Time": "date_time",
    "Temp_C": "temp_c",
    "Dew Point Temp_C": "dew_point_temp_c",
    "Rel Hum_%": "real_hum_pct",
    "Wind Speed_km/h": "wind_speed_kmh",
    "Visibility_km": "visibility_km",
    "Press_kPa": "press_kpa",
}


def simplify_weather_columns(df: pd.DataFrame) -> pd.DataFrame:

    df.rename(columns=NEW_COLUMN_NAMES, inplace=True)
    return df


def convert_date_to_datetime(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    date_column: str = kwargs.get("date_column", "date_time")
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def get_oscilations(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    return df[(df["ColumnX1"] == col1) & (df["ColumnX2"] == col2)]


def get_lag_of_causality_per_pair(
    df: pd.DataFrame, causality_column: str, is_max: bool = False
) -> pd.DataFrame:
    df.reset_index(inplace=True, drop=True)
    if is_max:
        idx = df.groupby(["ColumnX1", "ColumnX2"])[causality_column].idxmax()
    else:
        idx = df.groupby(["ColumnX1", "ColumnX2"])[causality_column].idxmin()

    df_grouped = df.loc[idx]
    return df_grouped


def add_from_and_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["from_column"] = np.where(df["Orientation"] == "from_1_to_2", df["ColumnX1"], df["ColumnX2"])
    df["to_column"] = np.where(df["Orientation"] == "from_1_to_2", df["ColumnX2"], df["ColumnX1"])
    return df


def convert_columns_in_column_to_human_readable(df: pd.DataFrame, column: str) -> pd.DataFrame:
    reversed_column_names = {v: k for k, v in NEW_COLUMN_NAMES.items()}
    df[column] = df[column].map(reversed_column_names)
    return df


def format_and_convert_to_latex(df: pd.DataFrame, causality_col: str) -> str:
    df_latex = df.copy(deep=True)
    df_latex = convert_columns_in_column_to_human_readable(df_latex, "from_column")
    df_latex = convert_columns_in_column_to_human_readable(df_latex, "to_column")
    df_latex.rename(columns={"from_column": "From Column", "to_column": "To Column"}, inplace=True)

    df_latex["Lag (days)"] = df_latex["Lag"] / 24

    for col in ["From Column", "To Column"]:
        df_latex[col] = df_latex[col].str.replace("_", " ")
        df_latex[col] = df_latex[col].str.replace("%", "\%")
        df_latex[col] = df_latex[col].str.replace("Point", "P.")
        df_latex[col] = df_latex[col].str.replace("Wind Speed", "WindSpeed")

    latex_table = df_latex[["From Column", "To Column", "Lag (days)", causality_col]].to_latex(
        index=False, header=True, float_format="%.2f"
    )
    return latex_table


def inplace_normalise_df(df: pd.DataFrame, column: str):
    df[f"{column}_normalized"] = (df[column] - df[column].min()) / (
        df[column].max() - df[column].min()
    )


def inplace_zscore_df(df: pd.DataFrame, column: str):
    df[f"{column}_zscore"] = (df[column] - df[column].mean()) / df[column].std()
