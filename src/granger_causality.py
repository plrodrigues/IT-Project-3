import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_granger_causality(data_array: np.ndarray, max_lag: int) -> list[float]:
    granger_causality = grangercausalitytests(data_array, max_lag)
    p_values = [
        round(granger_causality[lag][0]["ssr_ftest"][1], 20) for lag in range(1, (max_lag + 1))
    ]
    return p_values


def get_array_orientations(arr_x1_x2: np.ndarray, arr_x2_x1: np.ndarray) -> list[float]:
    if len(arr_x1_x2) != len(arr_x2_x1):
        raise NotImplementedError(
            "We can't compare arrays with different sizes. Please verify the source of information."
        )

    orientations = []
    for i in range(len(arr_x1_x2)):
        if arr_x1_x2[i] <= arr_x2_x1[i]:
            orientations.append("from_1_to_2")
        else:
            orientations.append("from_2_to_1")

    return orientations


def run_granger_causality_for_dataframe(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    df_result = pd.DataFrame()
    # Compare each 2 by 2
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            if i != j:
                col_x1 = numerical_columns[i]
                col_x2 = numerical_columns[j]
                array_p_values_per_lag_x1_to_x2 = get_granger_causality(
                    df[[col_x1, col_x2]].to_numpy(), max_lag
                )
                array_p_values_per_lag_x2_to_x1 = get_granger_causality(
                    df[[col_x2, col_x1]].to_numpy(), max_lag
                )

                # Detect the orientation
                array_of_orientations = get_array_orientations(
                    array_p_values_per_lag_x1_to_x2, array_p_values_per_lag_x2_to_x1
                )

                # Create a dictionary to store the result
                df_result_x = pd.DataFrame(
                    {
                        "ColumnX1": [col_x1] * len(array_of_orientations),
                        "ColumnX2": [col_x2] * len(array_of_orientations),
                        "Lag": range(max_lag),
                        "GrangerCausality": np.minimum(
                            array_p_values_per_lag_x1_to_x2, array_p_values_per_lag_x2_to_x1
                        ),
                        "Orientation": array_of_orientations,
                    }
                )
                # Append the result dictionary to the results list
                df_result = pd.concat([df_result, df_result_x])
    return df_result
