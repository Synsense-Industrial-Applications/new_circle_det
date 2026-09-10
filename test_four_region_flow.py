from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from four_region_flow import FEATURE_TO_DIRECTION, load_flow_csv


class FourRegionFlowTests(unittest.TestCase):
    def test_exact_feature_mapping_from_reference_player(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("x", "y", "feature", "timestamp"))
                for feature in range(8):
                    writer.writerow((10, 20, feature, 100 + feature))

            data = load_flow_csv(path)

        np.testing.assert_array_equal(data.x, [20, 20, 21, 21, 20, 20, 21, 21])
        np.testing.assert_array_equal(data.y, [40, 41, 40, 41, 40, 41, 40, 41])
        np.testing.assert_array_equal(data.direction, FEATURE_TO_DIRECTION)


if __name__ == "__main__":
    unittest.main()
