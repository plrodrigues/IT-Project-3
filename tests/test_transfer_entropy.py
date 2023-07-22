import unittest

import numpy as np
import pandas as pd

from src.transfer_entropy import get_transfer_entropy


class TestTransferEntropy(unittest.TestCase):
    @staticmethod
    def get_bistable_system_data() -> tuple[np.ndarray, np.ndarray]:
        # Bistable system example
        xs = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])  # Light switch 'X' (0: OFF, 1: ON)
        ys = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])  # Lightbulb 'Y' (0: OFF, 1: ON)
        return xs, ys

    @staticmethod
    def get_time_reversal_symmetry_data() -> tuple[np.ndarray, np.ndarray]:
        # Time-reversal symmetry example
        xs = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])  # 'X' has a periodic pattern
        ys = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1])  # 'Y' follows the same pattern as 'X'
        return xs, ys

    @staticmethod
    def get_invertible_switch_data() -> tuple[np.ndarray, np.ndarray]:
        xs = [0, 1, 1, 1, 1, 0, 0, 0, 0]
        ys = [0, 0, 1, 1, 1, 1, 0, 0, 0]
        return xs, ys

    def test_case_1_te_with_lag_2(self):
        xs, ys = self.get_bistable_system_data()
        # Test Case 1: Transfer entropy with lag 2
        te = get_transfer_entropy(target_column=ys, causal_column=xs, lag=2)
        self.assertAlmostEqual(te, 0.39355535745192394, places=6)

    def test_case_1_te_with_lag_3(self):
        xs, ys = self.get_bistable_system_data()
        # Test Case 1: Transfer entropy with lag 3
        te = get_transfer_entropy(target_column=ys, causal_column=xs, lag=3)
        self.assertEqual(te, 1.0)

    def test_case_2_te_with_lag_2(self):
        xs, ys = self.get_time_reversal_symmetry_data()
        # Test Case 2: Transfer entropy with lag 2
        te = get_transfer_entropy(target_column=ys, causal_column=xs, lag=2)
        self.assertAlmostEqual(te, 0.0, places=6)

    def test_case_3_te_with_lag_1_one_way(self):
        xs, ys = self.get_invertible_switch_data()
        # Test Case 3: Transfer entropy with lag 1
        te = get_transfer_entropy(target_column=ys, causal_column=xs, lag=1)
        self.assertAlmostEqual(te, 0.8112781244591329, places=6)

    def test_case_3_te_with_lag_1_inverted(self):
        xs, ys = self.get_invertible_switch_data()
        # Test Case 3: Transfer entropy with lag 1
        te = get_transfer_entropy(target_column=xs, causal_column=ys, lag=1)
        self.assertAlmostEqual(te, 0.12255624891826589, places=6)


if __name__ == "__main__":
    unittest.main()
