import unittest

import numpy as np
import pandas as pd

from src.transfer_entropy import transfer_entropy


class TestTransferEntropy(unittest.TestCase):
    @staticmethod
    def generate_test_data(length):
        """
        Generate test data with two columns for transfer entropy computation.

        Parameters:
            length (int): Length of the time series.

        Returns:
            pandas.DataFrame: DataFrame with 'x' and 'y' columns.
        """
        # Generate time series data
        np.random.seed(123)
        t = np.arange(length)
        x = np.sin(0.1 * t) + np.random.normal(0, 0.2, length)
        y = np.roll(x, 3) + np.random.normal(0, 0.2, length)

        # Create DataFrame
        df = pd.DataFrame({"x": x, "y": y})

        return df

    def test_case_1_te_with_lag_1(self):
        # Test Case 1: Transfer entropy with lag 1
        df = self.generate_test_data(100)
        te1 = get_shannons_entropy(df["x"], df["y"], history=1)
        print(f"Transfer Entropy (Lag 1): {te1}")

    def test_case_2_te_with_lat_2(self):
        # Test Case 2: Transfer entropy with lag 2
        df = self.generate_test_data(100)
        te2 = get_shannons_entropy(df["x"], df["y"], history=2)
        print(f"Transfer Entropy (Lag 2): {te2}")

    def test_case_3_te_with_lag3(self):
        # Test Case 3: Transfer entropy with lag 3
        df = self.generate_test_data(100)
        te3 = get_shannons_entropy(df["x"], df["y"], history=3)
        print(f"Transfer Entropy (Lag 3): {te3}")


if __name__ == "__main__":
    unittest.main()
