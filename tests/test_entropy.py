import unittest

import numpy as np

from src.entropy import (
    get_shannons_entropy_from_array,
    get_shannons_entropy_from_probabilities,
)


class EntropyTestCase(unittest.TestCase):
    def test_binary_sequence_entropy(self):
        binary_sequence = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
        entropy_score = get_shannons_entropy_from_array(binary_sequence)

        self.assertAlmostEqual(entropy_score, 0.9709505944546686, places=6)

    def test_categorical_data_entropy(self):
        categorical_data = ["A", "B", "C", "A", "B", "B", "C", "C", "C", "C"]
        entropy_score = get_shannons_entropy_from_array(categorical_data)

        self.assertAlmostEqual(entropy_score, 1.4854752972273344, places=6)

    def test_continuous_data_entropy(self):
        np.random.seed(123)
        continuous_data = np.random.normal(0, 1, size=1000)
        entropy_score = get_shannons_entropy_from_array(continuous_data)

        self.assertAlmostEqual(entropy_score, 5.845386843706611, places=6)

    def test_binary_event_fair_coin_entropy(self):
        # fair coin probability
        binary_event_probability = np.array([0.5, 0.5])
        entropy_score = get_shannons_entropy_from_probabilities(binary_event_probability)
        self.assertEqual(entropy_score, 1.0)

    def test_binary_event_bias_coin_entropy(self):
        # bias coin probability
        binary_event_probability = np.array([9 / 10, 1 / 10])
        entropy_score = get_shannons_entropy_from_probabilities(binary_event_probability)
        self.assertAlmostEqual(entropy_score, 0.46899559358928117, places=6)


if __name__ == "__main__":
    unittest.main()
