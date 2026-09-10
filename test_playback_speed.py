from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circle_detection.detector import FlowEvent
from four_region_flow import FlowData
from visualize_circle_detection import (
    DSCTEventPlayer,
    PLAYBACK_SPEED_PRESETS,
    make_configs,
    precompute_detections,
)


class PlaybackSpeedTests(unittest.TestCase):
    def test_playback_uses_source_timestamps_and_time_multiplier(self) -> None:
        count = 600
        angle = np.linspace(0.0, 4.0 * np.pi, count, endpoint=False)
        x = np.rint(64.0 + 26.0 * np.cos(angle)).astype(np.int16)
        y = np.rint(64.0 + 26.0 * np.sin(angle)).astype(np.int16)
        direction = (np.arange(count) % 4).astype(np.int8)
        timestamp = np.arange(count, dtype=np.int64) * 100
        data = FlowData(
            path=Path("synthetic.csv"),
            x=x,
            y=y,
            direction=direction,
            timestamp_us=timestamp,
            relative_s=timestamp.astype(float) * 1e-6,
            feature=np.zeros(count, dtype=np.int16),
            layer=np.zeros(count, dtype=np.int16),
        )
        events = [
            FlowEvent(float(px), float(py), int(code), float(t))
            for px, py, code, t in zip(x, y, direction, timestamp)
        ]
        strict, adaptive = make_configs()
        precomputed = precompute_detections(
            events, strict, adaptive, show_progress=False
        )
        with patch("visualize_circle_detection.time.perf_counter", return_value=10.0):
            player = DSCTEventPlayer(
                data,
                events,
                precomputed,
                trail_ms=20.0,
                fade_tau_ms=5.0,
                interval_ms=40,
                playback_speed=1.0,
                confidence_threshold=0.30,
                history_events=1500,
                enable_timer=False,
                confidence_metric="confidence",
            )
        try:
            # 20.05 ms of wall time at 1x selects the last event at 20,000 us.
            # The small offset avoids an artificial float boundary at exactly
            # 20,000 integer microseconds.
            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=10.02005
            ):
                player._timer_tick(0)
            self.assertEqual(player.index, 200)
            self.assertAlmostEqual(player.playhead_timestamp_us, 20_050.0)

            # Changing to 2x preserves the current source time. The following
            # 10 ms of wall time advances another 20 ms in the recording.
            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=10.02005
            ):
                player.speed_slider.set_val(PLAYBACK_SPEED_PRESETS.index(2.0))
            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=10.03005
            ):
                player._timer_tick(1)
            self.assertEqual(player.index, 400)
            self.assertAlmostEqual(player.playhead_timestamp_us, 40_050.0)
            active = player._active_slice()
            self.assertEqual(active.start, 201)
            self.assertEqual(active.stop, 401)

            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=10.03005
            ):
                player.progress_slider.set_val(500)
            self.assertEqual(player.index, 499)
            self.assertEqual(player.playhead_timestamp_us, timestamp[499])
            self.assertFalse(player.playing)
            self.assertIsNotNone(player.adaptive_detection)

            player.threshold_slider.set_val(0.99)
            self.assertIsNone(player.adaptive_detection)
            self.assertFalse(player.adaptive_circle.get_visible())

            # A selected event interval repeats according to its endpoint
            # timestamps. Here 11.05 ms from index 100 wraps 1.05 ms into the
            # 100..200 interval, landing on index 110.
            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=20.0
            ):
                player.loop_slider.set_val((101, 201))
                player._toggle_loop()
                player.speed_slider.set_val(PLAYBACK_SPEED_PRESETS.index(1.0))
                player._set_playing(True)
            self.assertTrue(player.loop_enabled)
            self.assertEqual(player.index, 100)

            with patch(
                "visualize_circle_detection.time.perf_counter", return_value=20.01105
            ):
                player._timer_tick(2)
            self.assertEqual(player.index, 110)
            self.assertTrue(player.playing)

            player.speed_slider.set_val(PLAYBACK_SPEED_PRESETS.index(0.01))
            self.assertEqual(player.playback_speed, 0.01)
        finally:
            plt.close(player.figure)


if __name__ == "__main__":
    unittest.main()
