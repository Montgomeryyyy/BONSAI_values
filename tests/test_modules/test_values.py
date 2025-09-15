import random
import unittest
from datetime import datetime
<<<<<<< HEAD

import numpy as np
import pandas as pd

from corebehrt.constants.data import CONCEPT_COL, VALUE_COL, VAL_TOKEN
from corebehrt.modules.features.values import ValueCreator


class TestValueCreator(unittest.TestCase):
=======
import pandas as pd

from corebehrt.modules.features.values import ValueCreator


class TestCreators(unittest.TestCase):
>>>>>>> 8a529b8e (more synth)
    def _generate_random_pids(self):
        while True:
            yield str(random.randint(1, 4))

    def _create_concepts(self, lab_dict):
        pids = self._generate_random_pids()
<<<<<<< HEAD
        rows = []
        for concept, values in lab_dict.items():
            for value in values:
                rows.append(
                    {
                        CONCEPT_COL: concept,
                        VALUE_COL: value,
                        "time": datetime.now(),
                        "subject_id": next(pids),
                    }
                )
        return pd.DataFrame(rows)

    def test_discrete_binning_without_prefix(self):
        concepts = self._create_concepts({"LAB1": [0.2, 0.3], "LAB2": [0.81, 0.42]})

        output = ValueCreator.add_values_discrete(
            concepts.copy(), bin_values=True, num_bins=10
        )

        sorted_codes = output.sort_values(["index", "order"])[CONCEPT_COL].tolist()
        expected_codes = [
            "LAB1",
            "VAL_2",
            "LAB1",
            "VAL_3",
            "LAB2",
            "VAL_8",
            "LAB2",
            "VAL_4",
        ]
        self.assertEqual(sorted_codes, expected_codes)

    def test_discrete_with_prefix_and_bin_mapping(self):
        concepts = self._create_concepts({"S/LAB1": [0.5], "L/LAB2": [1.0]})
        bin_mapping = {"S/LAB1": 4, "L/LAB2": 2}

        output = ValueCreator.add_values_discrete(
            concepts.copy(),
            bin_values=True,
            bin_mapping=bin_mapping,
            num_bins=10,
            add_prefix=True,
            separator_regex=r"^([^/]+)/",
        )

        sorted_codes = output.sort_values(["index", "order"])[CONCEPT_COL].tolist()
        expected_codes = ["S/LAB1", "S/VAL_2", "L/LAB2", "L/VAL_1"]
        self.assertEqual(sorted_codes, expected_codes)

    def test_discrete_with_prefix_default_bins(self):
        concepts = self._create_concepts({"S/LAB1": [0.25], "P/LAB3": [0.75]})

        output = ValueCreator.add_values_discrete(
            concepts.copy(),
            bin_values=True,
            num_bins=4,
            add_prefix=True,
            separator_regex=r"^([^/]+)/",
        )

        sorted_codes = output.sort_values(["index", "order"])[CONCEPT_COL].tolist()
        # num_bins=4 -> bins: 0.25 -> 1, 0.75 -> 3
        expected_codes = ["S/LAB1", "S/VAL_1", "P/LAB3", "P/VAL_3"]
        self.assertEqual(sorted_codes, expected_codes)
        self.assertNotIn(VALUE_COL, output.columns)

    def test_continuous_values_with_binning(self):
        concepts = self._create_concepts({"LAB1": [0.1, 0.9, np.nan]})

        output = ValueCreator.add_values_continuous(
            concepts.copy(), bin_values=True, num_bins=5
        ).sort_values(["index", "order"])

        codes = output[CONCEPT_COL].tolist()
        values = output[VALUE_COL].tolist()

        expected_codes = ["LAB1", VAL_TOKEN, "LAB1", VAL_TOKEN, "LAB1"]
        expected_values = [np.nan, 0, np.nan, 4, np.nan]

        self.assertEqual(codes, expected_codes)
        for actual, expected in zip(values, expected_values):
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertEqual(actual, expected)

    def test_continuous_values_without_binning(self):
        concepts = self._create_concepts({"LAB1": [0.1, 0.9, np.nan]})

        output = ValueCreator.add_values_continuous(
            concepts.copy(), bin_values=False
        ).sort_values(["index", "order"])

        codes = output[CONCEPT_COL].tolist()
        values = output[VALUE_COL].tolist()

        expected_codes = ["LAB1", VAL_TOKEN, "LAB1", VAL_TOKEN, "LAB1"]
        expected_values = [np.nan, 0.1, np.nan, 0.9, np.nan]

        self.assertEqual(codes, expected_codes)
        for actual, expected in zip(values, expected_values):
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertAlmostEqual(actual, expected)

    def test_bin_function_handles_strings_and_nan(self):
        values = pd.Series([0.0, "0.5", "invalid", np.nan, 1.0])

        binned = ValueCreator.bin(values, num_bins=10)

        self.assertEqual(binned.iloc[0], 0)
        self.assertEqual(binned.iloc[1], 5)
        self.assertTrue(pd.isna(binned.iloc[2]))
        self.assertTrue(pd.isna(binned.iloc[3]))
        # 1.0 is clamped just below 1 before multiplying by num_bins
        self.assertEqual(binned.iloc[4], 9)

    def test_add_values_router_discrete_and_continuous(self):
        concepts = self._create_concepts({"LAB1": [0.2], "LAB2": [0.6]})

        out_discrete = ValueCreator.add_values(
            concepts.copy(), value_type="discrete", bin_values=True, num_bins=5
        ).sort_values(["index", "order"])
        out_continuous = ValueCreator.add_values(
            concepts.copy(), value_type="combined", bin_values=False
        ).sort_values(["index", "order"])

        discrete_codes = out_discrete[CONCEPT_COL].tolist()
        continuous_codes = out_continuous[CONCEPT_COL].tolist()
        continuous_values = out_continuous[VALUE_COL].tolist()

        self.assertEqual(discrete_codes, ["LAB1", "VAL_1", "LAB2", "VAL_3"])
        self.assertEqual(continuous_codes, ["LAB1", VAL_TOKEN, "LAB2", VAL_TOKEN])
        expected_values = [np.nan, 0.2, np.nan, 0.6]
        for actual, expected in zip(continuous_values, expected_values):
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertAlmostEqual(actual, expected)

    def test_add_values_router_invalid_type(self):
        concepts = self._create_concepts({"LAB1": [0.2]})
        with self.assertRaises(ValueError):
            ValueCreator.add_values(concepts, value_type="xxx")
=======
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
        binned_values = ValueCreator.bin_results(self.concepts_pd_normed, num_bins=100)
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
        )
        expected_binned_concepts = [
            [
                lab,
                "VAL_" + str(int(float(value) * 100)) if value != "Kommentar" else None,
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
        binned_values = ValueCreator.bin_results(
            self.concepts_pd_normed_prefix,
            num_bins=100,
            add_prefix=True,
            separator_regex=r"^([^/]+)/",
        )
        sorted_concepts = list(
            binned_values.sort_values(by=["index", "order"]).sort_index()["code"]
        )

        expected_binned_concepts = []
        for lab, values in self.lab_dict_normed_prefix.items():
            prefix = lab.split("/")[0]  # Extract prefix from key
            for value in values:
                expected_binned_concepts.append(
                    [
                        lab,
                        f"{prefix}/VAL_{int(float(value) * 100)}"
                        if value != "Kommentar"
                        else None,
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
            binned_values = ValueCreator.bin_results(nan_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with all-NaN input: {e}")

    def test_empty_dataframe(self):
        """Test that the bin_results method handles empty DataFrames correctly."""
        # Create an empty DataFrame with only headers
        empty_df = pd.DataFrame(columns=["code", "numeric_value", "time", "subject_id"])

        # This should not raise an error
        try:
            binned_values = ValueCreator.bin_results(empty_df, num_bins=100)
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
            binned_values = ValueCreator.bin_results(single_nan_df, num_bins=100)
            self.assertIsInstance(binned_values, pd.DataFrame)
        except ValueError as e:
            self.fail(f"bin_results raised ValueError with single NaN value: {e}")
>>>>>>> 8a529b8e (more synth)


if __name__ == "__main__":
    unittest.main()
