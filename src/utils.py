import pandas as pd


def simplify_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_column_names = {
        "Weather": "weather",
        "Date/Time": "date_time",
        "Temp_C": "temp_c",
        "Dew Point Temp_C": "dew_point_temp_c",
        "Rel Hum_%": "real_hum_pct",
        "Wind Speed_km/h": "wind_speed_kmh",
        "Visibility_km": "visibility_km",
        "Press_kPa": "press_kpa",
    }
    df.rename(columns=new_column_names, inplace=True)
    return df


def convert_date_to_datetime(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    date_column: str = kwargs.get("date_column", "date_time")
    df[date_column] = pd.to_datetime(df[date_column])
    return df
