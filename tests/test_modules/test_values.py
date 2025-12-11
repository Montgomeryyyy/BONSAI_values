import random
import unittest
from datetime import datetime
import pandas as pd

from corebehrt.modules.features.values import ValueCreatorDiscrete


class TestCreators(unittest.TestCase):
    def _generate_random_pids(self):
        while True:
            yield str(random.randint(1, 4))

    def _create_concepts(self, lab_dict):
        pids = self._generate_random_pids()
        lab_df_list = []
        for concept, values in lab_dict.items():
            for value in values:
                lab_df_list.append(
                    {
                        "code": concept,
                        "numeric_value": value,
                        "time": pd.Timestamp.now(),
                        "subject_id": next(pids),
                    }
                )
        lab_df = pd.DataFrame(lab_df_list)
        return lab_df

    def setUp(self):
        self.origin_point = datetime(2020, 1, 26)
        self.background_vars = ["GENDER"]

        self.lab_dict_normed = {
            "LAB1": ["0.20", "0.30", "0.40", "0.25", "0.35", "0.21"],
            "LAB2": ["0.99", "0.81", "0.42", "0.21"],
            "LAB4": ["Kommentar"],
            "LAB5": ["0.11", "0.15", "0.12"],
            "LAB6": ["1.00"],
        }

        self.lab_dict_normed_prefix = {
            "S/LAB1": ["0.20", "0.30", "0.40", "0.25", "0.35", "0.21"],
            "S/LAB2": ["0.99", "0.81", "0.42", "0.21"],
            "L/LAB4": ["Kommentar"],
            "L/LAB5": ["0.11", "0.15", "0.12"],
            "L/LAB6": ["1.00"],
        }

        # Create sample data as pandas DataFrames
        self.concepts_pd_normed = self._create_concepts(self.lab_dict_normed)
        self.concepts_pd_normed_prefix = self._create_concepts(
            self.lab_dict_normed_prefix
        )

    def test_create_binned_value(self):
        binned_values = ValueCreatorDiscrete.bin_results(
            self.concepts_pd_normed, num_bins=100
        )
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
        )

        def calculate_bin(value_str, num_bins=100):
            """Calculate bin value matching the actual binning logic with clamping."""
            if value_str == "Kommentar":
                return None
            val = float(value_str)
            # Clamp to [0, 1.0 - 1e-10) to match the actual function
            val = max(0.0, min(val, 1.0 - 1e-10))
            bin_idx = int(val * num_bins)
            return "VAL_" + str(bin_idx)

        expected_binned_concepts = [
            [
                lab,
                calculate_bin(value, num_bins=100),
            ]
            for lab, values in self.lab_dict_normed.items()
            for value in values
        ]
        expected_flattened_binned_concepts = [
            item
            for sublist in expected_binned_concepts
            for item in sublist
            if item is not None
        ]
        self.assertEqual(sorted_concepts, expected_flattened_binned_concepts)

    def test_create_binned_value_with_prefix(self):
        binned_values = ValueCreatorDiscrete.bin_results(
            self.concepts_pd_normed_prefix,
            num_bins=100,
            add_prefix=True,
            separator_regex=r"^([^/]+)/",
        )
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
        )

        def calculate_bin(value_str, num_bins=100):
            """Calculate bin value matching the actual binning logic with clamping."""
            if value_str == "Kommentar":
                return None
            val = float(value_str)
            # Clamp to [0, 1.0 - 1e-10) to match the actual function
            val = max(0.0, min(val, 1.0 - 1e-10))
            bin_idx = int(val * num_bins)
            return "VAL_" + str(bin_idx)

        expected_binned_concepts = []
        for lab, values in self.lab_dict_normed_prefix.items():
            prefix = lab.split("/")[0]  # Extract prefix from key
            for value in values:
                bin_val = calculate_bin(value, num_bins=100)
                expected_binned_concepts.append(
                    [
                        lab,
                        f"{prefix}/{bin_val}" if bin_val is not None else None,
                    ]
                )
        expected_flattened_binned_concepts = [
            item
            for sublist in expected_binned_concepts
            for item in sublist
            if item is not None
        ]
        self.assertEqual(sorted_concepts, expected_flattened_binned_concepts)

    def test_all_nan_values(self):
        """Test that the bin_results method handles cases where all numeric values are NaN."""
        # Create a DataFrame with only NaN values in numeric_value
        nan_df = pd.DataFrame(
            {
                "code": ["LAB_NAN1", "LAB_NAN2", "LAB_NAN3"],
                "numeric_value": [float("nan"), float("nan"), float("nan")],
                "time": [pd.Timestamp.now()] * 3,
                "subject_id": ["1", "2", "3"],
            }
        )

        # This should not raise an error
        try:
            binned_values = ValueCreatorDiscrete.bin_results(nan_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with all-NaN input: {e}")

    def test_empty_dataframe(self):
        """Test that the bin_results method handles empty DataFrames correctly."""
        # Create an empty DataFrame with only headers
        empty_df = pd.DataFrame(columns=["code", "numeric_value", "time", "subject_id"])

        # This should not raise an error
        try:
            binned_values = ValueCreatorDiscrete.bin_results(empty_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with empty DataFrame: {e}")

    def test_single_row_nan(self):
        """Test that the bin_results method handles a single row with NaN."""
        # Create a DataFrame with a single row containing NaN
        single_nan_df = pd.DataFrame(
            {
                "code": ["LAB_SINGLE"],
                "numeric_value": [float("nan")],
                "time": [pd.Timestamp.now()],
                "subject_id": ["1"],
            }
        )

        # This should not raise an error
        try:
            binned_values = ValueCreatorDiscrete.bin_results(
                single_nan_df, num_bins=100
            )
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with single NaN value: {e}")

    # Tests for the bin() static method
    def test_bin_basic(self):
        """Test basic binning functionality."""
        values = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Check that all values are binned
        self.assertEqual(len(binned), len(values))
        # Check that binned values have VAL_ prefix
        self.assertTrue(all(str(v).startswith("VAL_") for v in binned if pd.notna(v)))
        # Check specific binning: 0.0 -> VAL_0, 0.25 -> VAL_25, 0.5 -> VAL_50, etc.
        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[1], "VAL_25")
        self.assertEqual(binned.iloc[2], "VAL_50")
        self.assertEqual(binned.iloc[3], "VAL_75")
        self.assertEqual(binned.iloc[4], "VAL_99")

    def test_bin_different_num_bins(self):
        """Test binning with different numbers of bins."""
        values = pd.Series([0.0, 0.5, 1.0])

        # Test with 10 bins
        # Calculation: clamp to [0, 1-1e-10), then value * num_bins, then convert to int
        # 0.0 * 10 = 0 -> VAL_0
        # 0.5 * 10 = 5.0 -> int(5.0) = 5 -> VAL_5
        # 1.0 clamped to ~0.9999999999, then * 10 = 9.999999999 -> int = 9 -> VAL_9
        binned_10 = ValueCreatorDiscrete.bin(values, num_bins=10)
        self.assertEqual(binned_10.iloc[0], "VAL_0")
        self.assertEqual(binned_10.iloc[1], "VAL_5")
        self.assertEqual(binned_10.iloc[2], "VAL_9")

        # Test with 4 bins
        # Calculation: clamp to [0, 1-1e-10), then value * num_bins, then convert to int
        # With num_bins=4, we get bins 0-3 (exactly 4 bins: 0 to num_bins-1)
        # 0.0 * 4 = 0 -> VAL_0
        # 0.5 * 4 = 2.0 -> int(2.0) = 2 -> VAL_2
        # 1.0 clamped to ~0.9999999999, then * 4 = 3.9999999996 -> int = 3 -> VAL_3
        binned_4 = ValueCreatorDiscrete.bin(values, num_bins=4)
        self.assertEqual(binned_4.iloc[0], "VAL_0")
        self.assertEqual(binned_4.iloc[1], "VAL_2")
        self.assertEqual(binned_4.iloc[2], "VAL_3")

        # Verify we get bins 0-3 (exactly 4 bins: 0 to num_bins-1)
        unique_bins = set(binned_4.dropna())
        self.assertEqual(len(unique_bins), 3)  # We have 3 test values
        # All bin indices should be < num_bins (0, 2, 3 are all < 4)
        self.assertTrue(all(int(b.replace("VAL_", "")) < 4 for b in unique_bins))

    def test_bin_with_nan(self):
        """Test binning with NaN values."""
        values = pd.Series([0.0, float("nan"), 0.5, float("nan"), 1.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Check that NaN values remain NaN
        self.assertTrue(pd.isna(binned.iloc[1]))
        self.assertTrue(pd.isna(binned.iloc[3]))
        # Check that non-NaN values are binned
        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[2], "VAL_50")
        self.assertEqual(binned.iloc[4], "VAL_99")

    def test_bin_all_nan(self):
        """Test binning with all NaN values."""
        values = pd.Series([float("nan"), float("nan"), float("nan")])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # All values should remain NaN
        self.assertTrue(binned.isna().all())

    def test_bin_empty_series(self):
        """Test binning with empty Series."""
        values = pd.Series([], dtype=float)
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Should return empty Series
        self.assertEqual(len(binned), 0)
        self.assertIsInstance(binned, pd.Series)

    def test_bin_boundary_values(self):
        """Test binning with boundary values (0.0 and 1.0)."""
        values = pd.Series([0.0, 1.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[1], "VAL_99")

    def test_bin_out_of_range_values(self):
        """Test binning with values outside [0, 1] range."""
        values = pd.Series([-0.1, 0.5, 1.5])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Values are clamped to [0, 1-1e-10) before binning
        # -0.1 clamped to 0.0 -> VAL_0
        # 0.5 -> VAL_50
        # 1.5 clamped to ~0.9999999999 -> VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[1], "VAL_50")
        self.assertEqual(binned.iloc[2], "VAL_99")

    def test_bin_callable_num_bins(self):
        """Test binning with callable num_bins function."""

        def custom_bins(n_unique):
            # Use more bins if there are more unique values
            return min(n_unique * 2, 200)

        values = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        binned = ValueCreatorDiscrete.bin(values, num_bins=custom_bins)

        # Should use custom binning based on unique count
        # 6 unique values -> 12 bins
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[5], "VAL_6")  # 0.5 * 12 = 6

    def test_bin_callable_with_duplicates(self):
        """Test callable num_bins with duplicate values."""

        def custom_bins(n_unique):
            return n_unique * 10

        # Only 3 unique values: 0.0, 0.5, 1.0
        values = pd.Series([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=custom_bins)

        # Should use 3 * 10 = 30 bins
        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_29
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[2], "VAL_15")  # 0.5 * 30 = 15
        self.assertEqual(binned.iloc[4], "VAL_29")  # 1.0 clamped then * 30 = 29

    def test_bin_string_values(self):
        """Test binning with string values that can be converted to numeric."""
        values = pd.Series(["0.0", "0.5", "1.0"])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Should convert strings to numeric and bin them
        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertEqual(binned.iloc[1], "VAL_50")
        self.assertEqual(binned.iloc[2], "VAL_99")

    def test_bin_non_convertible_strings(self):
        """Test binning with strings that cannot be converted to numeric."""
        values = pd.Series(["0.0", "invalid", "1.0"])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Non-convertible strings should become NaN
        # Note: 1.0 is clamped to (1.0 - 1e-10) before binning, so it becomes VAL_99
        self.assertEqual(binned.iloc[0], "VAL_0")
        self.assertTrue(pd.isna(binned.iloc[1]))
        self.assertEqual(binned.iloc[2], "VAL_99")

    def test_bin_single_value(self):
        """Test binning with a single value."""
        values = pd.Series([0.5])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        self.assertEqual(len(binned), 1)
        self.assertEqual(binned.iloc[0], "VAL_50")

    def test_bin_all_zeros(self):
        """Test binning with all zero values."""
        values = pd.Series([0.0, 0.0, 0.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # All should be binned to VAL_0
        self.assertTrue(all(b == "VAL_0" for b in binned))

    def test_bin_all_ones(self):
        """Test binning with all one values."""
        values = pd.Series([1.0, 1.0, 1.0])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # All should be binned to VAL_99 (1.0 is clamped to 1.0 - 1e-10 before binning)
        self.assertTrue(all(b == "VAL_99" for b in binned))

    def test_bin_very_small_values(self):
        """Test binning with very small values."""
        values = pd.Series([0.001, 0.0001, 0.00001])
        binned = ValueCreatorDiscrete.bin(values, num_bins=100)

        # Very small values should still be binned
        self.assertEqual(binned.iloc[0], "VAL_0")  # 0.001 * 100 = 0.1 -> 0
        self.assertEqual(binned.iloc[1], "VAL_0")  # 0.0001 * 100 = 0.01 -> 0
        self.assertEqual(binned.iloc[2], "VAL_0")  # 0.00001 * 100 = 0.001 -> 0


if __name__ == "__main__":
    unittest.main()
