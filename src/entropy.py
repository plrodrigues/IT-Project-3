import math

import numpy as np
from loguru import logger


def normalise_array(x: np.ndarray, bins: int) -> np.ndarray:
    x = [float(value) for value in x]
    x_norm = x / ((np.max(x) - np.min(x)) / bins)
    return x_norm


def get_marginal_probabilities(x: np.ndarray, **kwargs) -> np.ndarray:
    bins = kwargs.get("bins", min(len(x), 100))
    # Normalize the input array
    if np.issubdtype(type(x[0]), np.number):
        logger.debug(f"Numerical data {x[0]}")
        x_norm = normalise_array(x, bins)
        # Compute histogram counts
        counts, _ = np.histogram(x_norm, bins=bins)
        # Normalize the counts to obtain probabilities
        probabilities = counts / np.sum(counts)
    else:
        logger.debug(f"Non-numerical data {x[0]}")
        _, value_counts = np.unique(x, return_counts=True)
        probabilities = value_counts / np.sum(value_counts)

    return probabilities


def get_shannons_entropy_from_probabilities(x_prob: np.ndarray, **kwargs) -> float:
    base = kwargs.get("base", 2)
    # Compute Entropy: - sum(pxi * log(pxi))
    log = math.log
    if base == 2:
        log = math.log2
    elif base == 10:
        log = math.log10
    else:
        logger.warning("Using base e for log.")

    entropy = -np.sum([pxi * log(pxi) for pxi in x_prob if pxi > 0])
    return entropy


def get_shannons_entropy_from_array(x: np.ndarray, **kwargs) -> float:
    # Get marginal probabilities
    bins = kwargs.get("bins", min(len(x), 100))
    x_prob = get_marginal_probabilities(x, bins=bins)

    # Compute Entropy from the probabilities
    entropy = get_shannons_entropy_from_probabilities(x_prob, **kwargs)
    return entropy
