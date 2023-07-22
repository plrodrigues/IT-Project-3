import numpy as np

from src.conditional_mutual_information import (
    get_conditional_mutual_information,
    get_mutual_information_betwen_3_arrays,
)
from src.mutual_information import get_mutual_information


def get_transfer_entropy(
    target_column: np.ndarray, causal_column: np.ndarray, lag: int, **kwargs
) -> float | tuple[float, float, float, float, float]:
    is_verbose_with_mi = kwargs.get("is_verbose_with_mi", False)
    # Generate the lag for both columns
    # we know each element is 1 hour appart
    target_column_lag = target_column[:-lag]
    causal_column_lag = causal_column[:-lag]
    target_column_recent = target_column[lag:]

    # Calculate the conditional mutual information between the variables selected, with the respective lags
    transfer_entropy = get_conditional_mutual_information(
        about_x=target_column_recent,
        with_y=causal_column_lag,
        knowing_z=target_column_lag,
        **kwargs,
    )
    mi_target_and_its_past = get_mutual_information(target_column_recent, target_column_lag)
    mi_target_and_past_of_causal = get_mutual_information(target_column_recent, causal_column_lag)
    mi_target_past_and_past_of_causal = get_mutual_information(target_column_lag, causal_column_lag)
    mi_target_recent_past_and_past_of_causal = get_mutual_information_betwen_3_arrays(
        target_column_recent, target_column_lag, causal_column_lag
    )
    if is_verbose_with_mi:
        return (
            transfer_entropy,
            mi_target_and_its_past,
            mi_target_and_past_of_causal,
            mi_target_past_and_past_of_causal,
            mi_target_recent_past_and_past_of_causal,
        )
    else:
        return transfer_entropy
