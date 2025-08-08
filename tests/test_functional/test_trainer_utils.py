import unittest

import torch

from corebehrt.functional.trainer.collate import dynamic_padding
from corebehrt.functional.trainer.utils import (
    convert_epochs_to_steps,
    replace_steps_with_epochs,
)
from corebehrt.constants.data import VALUE_FEAT, VALUE_NULL_TOKEN


class TestTrainerUtils(unittest.TestCase):
    def test_convert_epochs_to_steps(self):
        # Test case 1: Basic conversion
        num_epochs = 2
        num_patients = 100
        batch_size = 10
        expected = 20  # (100 / 10) * 2

        self.assertEqual(
            convert_epochs_to_steps(num_epochs, num_patients, batch_size), expected
        )

        # Test case 2: Edge case with small batch size
        num_epochs = 1
        num_patients = 5
        batch_size = 2
        expected = 3  # ceil(5 / 2) * 1

        self.assertEqual(
            convert_epochs_to_steps(num_epochs, num_patients, batch_size), expected
        )

    def test_replace_steps_with_epochs(self):
        # Test case 1: Basic replacement
        config = {"steps": 100, "num_patients": 1000, "batch_size": 10}
        expected_epochs = 1  # 100 / (1000 / 10)

        result = replace_steps_with_epochs(config)
        self.assertEqual(result["epochs"], expected_epochs)
        self.assertNotIn("steps", result)

        # Test case 2: Edge case with small batch size
        config = {"steps": 5, "num_patients": 10, "batch_size": 3}
        expected_epochs = 2  # 5 / ceil(10 / 3) = 5 / 4 = 1.25 -> 2

        result = replace_steps_with_epochs(config)
        self.assertEqual(result["epochs"], expected_epochs)

    def test_convert_epochs_to_steps_multiple(self):
        # Test multiple configurations
        test_cases = [
            (1, 100, 10, 10),  # 1 epoch, 100 patients, batch 10 -> 10 steps
            (2, 100, 10, 20),  # 2 epochs, 100 patients, batch 10 -> 20 steps
            (1, 50, 10, 5),    # 1 epoch, 50 patients, batch 10 -> 5 steps
            (3, 150, 15, 30),  # 3 epochs, 150 patients, batch 15 -> 30 steps
        ]

        for num_epochs, num_patients, batch_size, expected in test_cases:
            with self.subTest(
                num_epochs=num_epochs,
                num_patients=num_patients,
                batch_size=batch_size,
            ):
                self.assertEqual(
                    convert_epochs_to_steps(num_epochs, num_patients, batch_size),
                    expected,
                )

    def test_dynamic_padding(self):
        # Test case 1: Padding sequences of different lengths
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor([1]),
            },
            {
                "concept": torch.tensor([1, 2, 3, 4, 5]),
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "target": torch.tensor([0]),
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 5))
        self.assertEqual(padded_batch["input_ids"].shape, (2, 5))
        self.assertEqual(padded_batch["attention_mask"].shape, (2, 5))
        self.assertEqual(padded_batch["target"].shape, (2, 1))

        # Check padding values
        self.assertTrue(
            torch.equal(padded_batch["input_ids"][0], torch.tensor([1, 2, 3, 0, 0]))
        )
        self.assertTrue(
            torch.equal(
                padded_batch["attention_mask"][0], torch.tensor([1, 1, 1, 0, 0])
            )
        )

        # Test case 2: Non-sequence tensors should remain unchanged
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "static_feature": torch.tensor([1.0, 2.0, 3.0]),
                "target": torch.tensor([1]),
            },
            {
                "concept": torch.tensor([1, 2, 3]),
                "static_feature": torch.tensor([4.0, 5.0, 6.0]),
                "target": torch.tensor([0]),
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 3))
        self.assertEqual(padded_batch["static_feature"].shape, (2, 3))
        self.assertEqual(padded_batch["target"].shape, (2, 1))

        # Check values
        self.assertTrue(
            torch.equal(
                padded_batch["static_feature"],
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            )
        )

        self.assertTrue(torch.equal(padded_batch["target"], torch.tensor([[1], [0]])))

    def test_dynamic_padding_with_values(self):
        # Test case: Padding sequences with VALUE_FEAT field
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                VALUE_FEAT: torch.tensor([1.5, 2.5, 3.5], dtype=torch.float),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor([1]),
            },
            {
                "concept": torch.tensor([1, 2, 3, 4, 5]),
                VALUE_FEAT: torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5], dtype=torch.float),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "target": torch.tensor([0]),
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 5))
        self.assertEqual(padded_batch[VALUE_FEAT].shape, (2, 5))
        self.assertEqual(padded_batch["attention_mask"].shape, (2, 5))
        self.assertEqual(padded_batch["target"].shape, (2, 1))

        # Check padding values for VALUE_FEAT
        self.assertTrue(
            torch.equal(
                padded_batch[VALUE_FEAT][0], 
                torch.tensor([1.5, 2.5, 3.5, VALUE_NULL_TOKEN, VALUE_NULL_TOKEN], dtype=torch.float)
            )
        )
        self.assertTrue(
            torch.equal(
                padded_batch[VALUE_FEAT][1], 
                torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5], dtype=torch.float)
            )
        )

        # Check padding values for other fields
        self.assertTrue(
            torch.equal(padded_batch["concept"][0], torch.tensor([1, 2, 3, 0, 0]))
        )
        self.assertTrue(
            torch.equal(
                padded_batch["attention_mask"][0], torch.tensor([1, 1, 1, 0, 0])
            )
        )

    def test_dynamic_padding_mlm_target(self):
        batch = [
            {
                "concept": torch.tensor([1, 2, 3]),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor([1, 0, 1]),  # Same length as input sequence
            },
            {
                "concept": torch.tensor([1, 2, 3, 4, 5]),
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "target": torch.tensor(
                    [0, 1, 1, 0, 1]
                ),  # Same length as input sequence
            },
        ]

        padded_batch = dynamic_padding(batch)

        # Check shapes
        self.assertEqual(padded_batch["concept"].shape, (2, 5))
        self.assertEqual(padded_batch["input_ids"].shape, (2, 5))
        self.assertEqual(padded_batch["attention_mask"].shape, (2, 5))
        self.assertEqual(
            padded_batch["target"].shape, (2, 5)
        )  # Target should be padded to max length

        # Check padding values
        self.assertTrue(
            torch.equal(
                padded_batch["target"][0],
                torch.tensor([1, 0, 1, -100, -100]),  # Padded with zeros
            )
        )
        self.assertTrue(
            torch.equal(
                padded_batch["target"][1],
                torch.tensor([0, 1, 1, 0, 1]),  # Original values unchanged
            )
        )


if __name__ == "__main__":
    unittest.main()
