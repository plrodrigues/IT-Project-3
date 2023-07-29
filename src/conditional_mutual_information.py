import numpy as np
from loguru import logger

from src.entropy import get_shannons_entropy_from_array, normalise_array
from src.mutual_information import (
    get_joint_entropy_from_2_arrays,
    get_joint_entropy_from_probability,
    get_mutual_information,
)


def get_joint_probabilities_of_3_arrays(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs
) -> np.ndarray:
    bins = kwargs.get("bins", min(len(x), 100))

    # Normalize the input arrays if they contain numerical data
    if np.issubdtype(type(x[0]), np.number):
        x_norm = normalise_array(x, bins)
        y_norm = normalise_array(y, bins)
        z_norm = normalise_array(z, bins)
        # Calculate the joint histogram counts
        counts, _ = np.histogramdd([x_norm, y_norm, z_norm], bins=[bins, bins, bins])
        # Normalize the counts to obtain probabilities
        probabilities = counts / np.sum(counts)
    else:
        _, x_value_counts = np.unique(x, return_counts=True)
        _, y_value_counts = np.unique(y, return_counts=True)
        _, z_value_counts = np.unique(z, return_counts=True)
        probabilities = np.zeros((len(x_value_counts), len(y_value_counts), len(z_value_counts)))
        for i, xi in enumerate(x_value_counts):
            for j, yj in enumerate(y_value_counts):
                for k, zk in enumerate(z_value_counts):
                    probabilities[i, j, k] = (xi * yj * zk) / np.sum(x_value_counts)

    return probabilities


def get_joint_entropy_from_3_arrays(x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs) -> float:
    # Get the joint probability
    xyz_joint_prob = get_joint_probabilities_of_3_arrays(x, y, z, **kwargs)
    # Calculate the joint entropy
    joint_entropy = get_joint_entropy_from_probability(xyz_joint_prob, **kwargs)
    return joint_entropy


def get_mutual_information_betwen_3_arrays(x, y, z, **kwargs) -> float:
    # Entropies x, y and z
    entropy_x = get_shannons_entropy_from_array(x, **kwargs)
    entropy_y = get_shannons_entropy_from_array(y, **kwargs)
    entropy_z = get_shannons_entropy_from_array(z, **kwargs)
    # Joint entropies for each pair
    joint_entropy_x_y = get_joint_entropy_from_2_arrays(x, y, **kwargs)
    joint_entropy_x_z = get_joint_entropy_from_2_arrays(x, z, **kwargs)
    joint_entropy_y_z = get_joint_entropy_from_2_arrays(y, z, **kwargs)
    # Joint entropy of the 3 variables
    joint_entropy_x_y_z = get_joint_entropy_from_3_arrays(x, y, z, **kwargs)

    # Combined mutual information
    mi_x_y_z = (
        entropy_x
        + entropy_y
        + entropy_z
        - joint_entropy_x_y
        - joint_entropy_x_z
        - joint_entropy_y_z
        + joint_entropy_x_y_z
    )
    return mi_x_y_z


def get_conditional_mutual_information(
    about_x: np.ndarray, with_y: np.ndarray, knowing_z: np.ndarray, **kwargs
) -> float:
    # Mutual information of every pair of variables
    mi_x_y = get_mutual_information(about_x, with_y, **kwargs)
    # Mutual information of the 3 variables
    mi_x_y_z = get_mutual_information_betwen_3_arrays(about_x, with_y, knowing_z, **kwargs)

    # Conditional mutual information
    cmi = mi_x_y - mi_x_y_z
    return cmi
