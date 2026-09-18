from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from offline_tools.event_stream_player import (
    CHANNEL_GROUP_ALL,
    CHANNEL_GROUP_HIGH,
    CHANNEL_GROUP_LOW,
    FEATURE_TO_DIRECTION as PLAYER_FEATURE_TO_DIRECTION,
    build_event_rgb,
    channel_group_mask,
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

            timestamp, x, y, direction, feature, source_shape = load_event_csv(path)

        np.testing.assert_array_equal(timestamp, np.arange(100, 116))
        np.testing.assert_array_equal(x, [20, 20, 21, 21] * 4)
        np.testing.assert_array_equal(y, [40, 41, 40, 41] * 4)
        np.testing.assert_array_equal(direction, PLAYER_FEATURE_TO_DIRECTION)
        np.testing.assert_array_equal(feature, np.arange(16))
        self.assertEqual(source_shape, "64x64x16 -> 128x128 four-direction flow")

    def test_player_can_filter_the_two_eight_channel_banks(self) -> None:
        features = np.arange(16, dtype=np.int16)
        np.testing.assert_array_equal(
            channel_group_mask(features, CHANNEL_GROUP_ALL),
            np.ones(16, dtype=bool),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(channel_group_mask(features, CHANNEL_GROUP_LOW)),
            np.arange(8),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(channel_group_mask(features, CHANNEL_GROUP_HIGH)),
            np.arange(8, 16),
        )
        with self.assertRaises(ValueError):
            channel_group_mask(features, "invalid")

    def test_render_count_follows_the_selected_channel_bank(self) -> None:
        timestamps = np.arange(4, dtype=np.int64) * 100
        xs = np.asarray([10, 20, 30, 40], dtype=np.int16)
        ys = np.asarray([12, 22, 32, 42], dtype=np.int16)
        directions = np.arange(4, dtype=np.int8)
        features = np.asarray([0, 7, 8, 15], dtype=np.int16)

        all_rgb, all_count = build_event_rgb(
            timestamps, xs, ys, directions, 3, 300, 1000, 500, 1.0,
            features, CHANNEL_GROUP_ALL,
        )
        low_rgb, low_count = build_event_rgb(
            timestamps, xs, ys, directions, 3, 300, 1000, 500, 1.0,
            features, CHANNEL_GROUP_LOW,
        )
        high_rgb, high_count = build_event_rgb(
            timestamps, xs, ys, directions, 3, 300, 1000, 500, 1.0,
            features, CHANNEL_GROUP_HIGH,
        )

        self.assertEqual((all_count, low_count, high_count), (4, 2, 2))
        self.assertFalse(np.array_equal(low_rgb, high_rgb))
        self.assertFalse(np.array_equal(all_rgb, low_rgb))


if __name__ == "__main__":
    unittest.main()
