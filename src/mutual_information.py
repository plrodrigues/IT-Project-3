import math

import numpy as np
from loguru import logger

from src.entropy import get_shannons_entropy_from_array, normalise_array


def get_joint_probabilities_of_numerical_numerical(
    x: np.ndarray, y: np.ndarray, bins: int
) -> np.ndarray:
    logger.debug(f"Numerical data {x[0]} and {y[0]}")
    x_norm = normalise_array(x, bins)
    y_norm = normalise_array(y, bins)
    # Calculate the joint histogram counts
    counts, _, _ = np.histogram2d(x_norm, y_norm, bins=bins)
    # Normalize the counts to obtain probabilities
    probabilities = counts / np.sum(counts)
    return probabilities


def get_joint_probabilities_of_numerical_categorical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logger.debug(f"Numerical data {x[0]} and categorical data {y[0]}")
    y_values, _ = np.unique(y, return_counts=True)
    probabilities = np.zeros((len(x), len(y_values)))
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y_values):
            probabilities[i, j] = np.sum((x == x_val) & (y == y_val))
    # Normalize the counts to obtain probabilities
    probabilities /= np.sum(probabilities)
    return probabilities


def get_joint_probabilities_of_categorical_numerical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logger.debug(f"Categorical data {x[0]} and numerical data {y[0]}")
    x_values, _ = np.unique(x, return_counts=True)
    probabilities = np.zeros((len(x_values), len(y)))
    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y):
            probabilities[i, j] = np.sum((x == x_val) & (y == y_val))
    # Normalize the counts to obtain probabilities
    probabilities /= np.sum(probabilities)
    return probabilities


def get_joint_probabilities_of_categorical_categorical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logger.debug(f"Categorical data {x[0]} and {y[0]}")
    x_values, _ = np.unique(x, return_counts=True)
    y_values, _ = np.unique(y, return_counts=True)
    probabilities = np.zeros((len(x_values), len(y_values)))
    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            probabilities[i, j] = np.sum((x == x_val) & (y == y_val))
    # Normalize the counts to obtain probabilities
    probabilities /= np.sum(probabilities)
    return probabilities


def get_joint_probabilities_of_2(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    bins = kwargs.get("bins", min(len(x), 100))

    if np.issubdtype(type(x[0]), np.number) and np.issubdtype(type(y[0]), np.number):
        return get_joint_probabilities_of_numerical_numerical(x, y, bins)
    elif np.issubdtype(type(x[0]), np.number) and not np.issubdtype(type(y[0]), np.number):
        return get_joint_probabilities_of_numerical_categorical(x, y)
    elif not np.issubdtype(type(x[0]), np.number) and np.issubdtype(type(y[0]), np.number):
        return get_joint_probabilities_of_categorical_numerical(x, y)
    else:
        return get_joint_probabilities_of_categorical_categorical(x, y)


def get_joint_entropy_from_probability(joint_probabilities: np.ndarray, **kwargs) -> float:
    base = kwargs.get("base", 2)
    # Filter out probabilities that are greater than 0
    non_zero_probs = joint_probabilities[joint_probabilities > 0]
    # Calculate the joint entropy
    log = np.log
    if base == 2:
        log = np.log2
    elif base == 10:
        log = np.log10
    else:
        logger.warning("Using base e for log.")

    joint_entropy = -np.sum(non_zero_probs * log(non_zero_probs))
    return joint_entropy


def get_joint_entropy_from_2_arrays(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    # Get the joint probability
    joint_probabilities = get_joint_probabilities_of_2(x, y, **kwargs)
    # Joint entropy
    joint_entropy = get_joint_entropy_from_probability(joint_probabilities, **kwargs)
    return joint_entropy


def get_mutual_information(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    # Calculate the marginal probabilities for x and y
    entropy_x = get_shannons_entropy_from_array(x, **kwargs)
    entropy_y = get_shannons_entropy_from_array(y, **kwargs)
    joint_entropy_x_y = get_joint_entropy_from_2_arrays(x, y, **kwargs)
    mutual_info = entropy_x + entropy_y - joint_entropy_x_y

    return mutual_info
