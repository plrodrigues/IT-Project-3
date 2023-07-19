import math

import numpy as np
from loguru import logger

from src.mutual_information import get_joint_probabilities_of_2


def calculate_conditional_entropy(
    x_joint_prob: np.ndarray, x_cond_prob: np.ndarray, **kwargs
) -> float:
    base = kwargs.get("base", 2)
    # Filter out probabilities that are greater than 0
    non_zero_probs = x_joint_prob[x_joint_prob > 0]
    # Calculate the conditional entropy
    log = np.log
    if base == 2:
        log = np.log2
    elif base == 10:
        log = np.log10
    else:
        logger.warning("Using base e for log.")
    entropy = -np.sum(non_zero_probs * log(x_cond_prob[x_cond_prob > 0]))
    return entropy


def get_conditional_mutual_information(
    x_prob: np.ndarray, y_prob: np.ndarray, z_prob: np.ndarray, **kwargs
) -> float:
    # Calculate the joint probabilities of X, Y, and Z
    xyz_joint_prob = get_joint_probabilities_of_2(x_prob, y_prob, **kwargs)
    xz_joint_prob = get_joint_probabilities_of_2(x_prob, z_prob, **kwargs)

    # Calculate the conditional probabilities p(x | z) and p(y | z)
    x_cond_prob = np.sum(xyz_joint_prob, axis=1) / np.sum(xz_joint_prob, axis=1)

    # Calculate the conditional entropy of X given Z
    h_x_given_z = calculate_conditional_entropy(xyz_joint_prob, x_cond_prob, **kwargs)

    # Calculate the conditional probabilities p(y | z)
    y_cond_prob = np.sum(xyz_joint_prob, axis=0) / np.sum(xz_joint_prob, axis=0)

    # Calculate the conditional entropy of Y given Z
    h_y_given_z = calculate_conditional_entropy(xyz_joint_prob.T, y_cond_prob, **kwargs)

    # Calculate the joint conditional entropy of X and Y given Z
    h_xy_given_z = calculate_conditional_entropy(xyz_joint_prob, xyz_joint_prob.flatten(), **kwargs)

    # Calculate the conditional mutual information
    cmi = h_x_given_z + h_y_given_z - h_xy_given_z
    return cmi
