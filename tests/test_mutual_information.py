import unittest

import numpy as np
from sklearn.metrics import mutual_info_score

from src.mutual_information import (
    get_joint_entropy_from_probability,
    get_joint_probabilities_of_2,
    get_mutual_information,
)


class MutualInformationTests(unittest.TestCase):
    def test_get_joint_probabilities(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        joint_prob = get_joint_probabilities_of_2(x, y, bins=3)
        expected_joint_prob = np.array([[0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.4]])

        np.testing.assert_allclose(joint_prob, expected_joint_prob, atol=1e-6)

    def test_get_joint_entropy(self):
        joint_probabilities = np.array(
            [
                [0.2, 0.1, 0.0],
                [0.0, 0.3, 0.1],
                [0.1, 0.0, 0.2],
            ]
        )
        joint_entropy = get_joint_entropy_from_probability(joint_probabilities)

        # Calculate the expected joint entropy manually
        expected_entropy = (
            -0.2 * np.log2(0.2)
            - 0.1 * np.log2(0.1)
            - 0.3 * np.log2(0.3)
            - 0.1 * np.log2(0.1)
            - 0.1 * np.log2(0.1)
            - 0.2 * np.log2(0.2)
        )

        # Assert that the calculated joint entropy matches the expected value
        self.assertAlmostEqual(joint_entropy, expected_entropy)

    def test_get_mutual_information(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        mi = get_mutual_information(x, y, bins=3)

        self.assertAlmostEqual(mi, 1.5219280948873621)

    def test_get_mutual_information_with_sklearn(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sklearn_mi = mutual_info_score(a, x)
        internal_mi = get_mutual_information(a, x, base="e")
        self.assertAlmostEqual(internal_mi, sklearn_mi, places=6)


if __name__ == "__main__":
    unittest.main()
