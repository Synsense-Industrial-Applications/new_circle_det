from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from offline_tools.event_stream_player import (
    FEATURE_TO_DIRECTION as PLAYER_FEATURE_TO_DIRECTION,
    load_event_csv,
)
from offline_tools.four_region_flow import FEATURE_TO_DIRECTION, load_flow_csv


class FourRegionFlowTests(unittest.TestCase):
    def test_exact_feature_mapping_from_reference_player(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("x", "y", "feature", "timestamp"))
                for feature in range(16):
                    writer.writerow((10, 20, feature, 100 + feature))

            data = load_flow_csv(path)

        np.testing.assert_array_equal(data.x, [20, 20, 21, 21] * 4)
        np.testing.assert_array_equal(data.y, [40, 41, 40, 41] * 4)
        np.testing.assert_array_equal(data.direction, FEATURE_TO_DIRECTION)

    def test_lightweight_player_uses_the_same_16_channel_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("x", "y", "feature", "timestamp"))
                for feature in range(16):
                    writer.writerow((10, 20, feature, 100 + feature))

            timestamp, x, y, direction, source_shape = load_event_csv(path)

        np.testing.assert_array_equal(timestamp, np.arange(100, 116))
        np.testing.assert_array_equal(x, [20, 20, 21, 21] * 4)
        np.testing.assert_array_equal(y, [40, 41, 40, 41] * 4)
        np.testing.assert_array_equal(direction, PLAYER_FEATURE_TO_DIRECTION)
        self.assertEqual(source_shape, "64x64x16 -> 128x128 four-direction flow")


if __name__ == "__main__":
    unittest.main()
