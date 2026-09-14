from __future__ import annotations

import math
import random
import unittest

import numpy as np

from circle_detection.adaptive_detector import (
    AdaptiveCircleDetector,
    AdaptiveDetectorConfig,
)
from circle_detection.detector import FlowEvent


class AdaptiveCircleDetectorTests(unittest.TestCase):
    def make_detector(self, **changes) -> AdaptiveCircleDetector:
        values = dict(
            window_events=300,
            min_events=80,
            hypotheses=500,
            update_interval_events=1,
            min_radius_px=4.0,
            max_radius_px=80.0,
            seed=91,
        )
        values.update(changes)
        return AdaptiveCircleDetector(AdaptiveDetectorConfig(**values))

    def test_recovers_circle_when_direction_codes_are_not_radial(self) -> None:
        rng = random.Random(17)
        detector = self.make_detector()
        cx, cy, radius = 61.5, 70.0, 30.0
        events = []
        for index in range(225):
            angle = rng.random() * 2.0 * math.pi
            events.append(
                FlowEvent(
                    cx + radius * math.cos(angle) + rng.gauss(0.0, 0.55),
                    cy + radius * math.sin(angle) + rng.gauss(0.0, 0.55),
                    rng.randrange(4),
                    float(index * index * 100),
                )
            )
        for index in range(75):
            events.append(
                FlowEvent(
                    rng.uniform(0.0, 127.0),
                    rng.uniform(0.0, 127.0),
                    rng.randrange(4),
                    float(10**9 + index),
                )
            )
        rng.shuffle(events)
        detector.extend(events)
        result = detector.detect(force=True)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(math.hypot(result.cx - cx, result.cy - cy), 1.5)
        self.assertLess(abs(result.radius - radius), 1.0)

    def test_batched_refinement_matches_scalar_refinement(self) -> None:
        detector = self.make_detector()
        angle = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        x = 62.0 + 24.0 * np.cos(angle)
        y = 69.0 + 24.0 * np.sin(angle)
        weights = np.exp(-np.arange(179, -1, -1, dtype=float) / 120.0)
        initial_cx = np.asarray([60.5, 63.4, 61.7])
        initial_cy = np.asarray([68.0, 70.2, 67.8])
        initial_radius = np.asarray([23.0, 25.2, 24.7])

        expected = np.asarray(
            [
                detector._refine(x, y, weights, cx, cy, radius)
                for cx, cy, radius in zip(
                    initial_cx, initial_cy, initial_radius
                )
            ]
        )
        actual = np.column_stack(
            detector._refine_batch(
                x,
                y,
                weights,
                initial_cx,
                initial_cy,
                initial_radius,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_rejects_straight_cross(self) -> None:
        rng = random.Random(8)
        detector = self.make_detector()
        for index in range(150):
            detector.push(FlowEvent(32.0, rng.uniform(0.0, 127.0), 0, float(index)))
            detector.push(FlowEvent(rng.uniform(0.0, 127.0), 64.0, 2, float(index) + 0.1))
        self.assertIsNone(detector.detect(force=True))

    def test_rejects_uniform_noise(self) -> None:
        rng = random.Random(57)
        detector = self.make_detector(hypotheses=700)
        for index in range(300):
            detector.push(
                FlowEvent(
                    rng.uniform(0.0, 127.0),
                    rng.uniform(0.0, 127.0),
                    rng.randrange(4),
                    float(index),
                )
            )
        self.assertIsNone(detector.detect(force=True))

    def test_rejects_border_l_shape_as_a_large_circle(self) -> None:
        rng = random.Random(77)
        detector = self.make_detector(hypotheses=2000, max_radius_px=90.0)
        for index in range(150):
            detector.push(
                FlowEvent(
                    rng.uniform(20.0, 127.0),
                    20.0 + rng.gauss(0.0, 0.5),
                    rng.randrange(4),
                    float(index),
                )
            )
        for index in range(150):
            detector.push(
                FlowEvent(
                    4.0 + rng.gauss(0.0, 0.5),
                    rng.uniform(20.0, 127.0),
                    rng.randrange(4),
                    float(150 + index),
                )
            )
        self.assertIsNone(detector.detect(force=True))

    def test_uses_event_count_not_timestamp_span(self) -> None:
        rng = random.Random(26)
        detector = self.make_detector(hypotheses=600)
        cx, cy, radius = 88.0, 45.0, 19.0
        timestamp = 0.0
        for index in range(260):
            angle = rng.random() * 2.0 * math.pi
            timestamp += 1.0 if index % 3 else 1_000_000.0
            detector.push(
                FlowEvent(
                    cx + radius * math.cos(angle) + rng.gauss(0.0, 0.4),
                    cy + radius * math.sin(angle) + rng.gauss(0.0, 0.4),
                    rng.randrange(4),
                    timestamp,
                )
            )
        result = detector.detect(force=True)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(math.hypot(result.cx - cx, result.cy - cy), 1.0)
        self.assertLess(abs(result.radius - radius), 0.8)

    def test_rejects_partial_top_arc_without_full_circle_support(self) -> None:
        rng = random.Random(109)
        detector = self.make_detector(hypotheses=1400)
        cx, cy, radius = 63.0, 67.0, 25.0
        for index in range(220):
            # Only 130 degrees of the top of the ball remain visible, which is
            # intentionally insufficient for the full-circle rules.
            angle = rng.uniform(math.radians(-155.0), math.radians(-25.0))
            detector.push(
                FlowEvent(
                    cx + radius * math.cos(angle) + rng.gauss(0.0, 0.35),
                    cy + radius * math.sin(angle) + rng.gauss(0.0, 0.35),
                    rng.randrange(4),
                    float(index),
                )
            )
        self.assertIsNone(detector.detect(force=True))

    def test_rejects_partial_lower_arc_without_full_circle_support(self) -> None:
        rng = random.Random(109)
        detector = self.make_detector(hypotheses=1400)
        cx, cy, radius = 63.0, 67.0, 25.0
        for index in range(220):
            angle = rng.uniform(math.radians(25.0), math.radians(155.0))
            detector.push(
                FlowEvent(
                    cx + radius * math.cos(angle) + rng.gauss(0.0, 0.35),
                    cy + radius * math.sin(angle) + rng.gauss(0.0, 0.35),
                    rng.randrange(4),
                    float(index),
                )
            )
        self.assertIsNone(detector.detect(force=True))

    def test_far_circle_requires_three_consistent_updates(self) -> None:
        rng = random.Random(3)
        detector = self.make_detector(
            window_events=120,
            min_events=60,
            hypotheses=1200,
            decay_events=1000.0,
            min_radius_px=5.0,
            max_radius_px=60.0,
            seed=3,
            jump_confirmations=3,
            track_reset_updates=3,
        )

        def push_circle(cx: float, cy: float, radius: float, count: int, start: int) -> None:
            for offset in range(count):
                angle = rng.random() * 2.0 * math.pi
                detector.push(
                    FlowEvent(
                        cx + radius * math.cos(angle) + rng.gauss(0.0, 0.25),
                        cy + radius * math.sin(angle) + rng.gauss(0.0, 0.25),
                        rng.randrange(4),
                        float(start + offset),
                    )
                )

        push_circle(35.0, 50.0, 15.0, 120, 0)
        first = detector.detect(force=True)
        self.assertIsNotNone(first)

        # Replace the complete event window with a far, much larger circle.
        # The first two estimates must be suppressed instead of producing a
        # one-frame teleport/size spike; the third consistent estimate may
        # start a confirmed new track.
        push_circle(95.0, 75.0, 28.0, 120, 1000)
        self.assertIsNone(detector.detect(force=True))
        push_circle(95.0, 75.0, 28.0, 6, 2000)
        self.assertIsNone(detector.detect(force=True))
        push_circle(95.0, 75.0, 28.0, 6, 2010)
        confirmed = detector.detect(force=True)
        self.assertIsNotNone(confirmed)
        assert confirmed is not None
        self.assertLess(math.hypot(confirmed.cx - 95.0, confirmed.cy - 75.0), 1.0)
        self.assertLess(abs(confirmed.radius - 28.0), 0.8)

    def test_relaxed_evidence_can_continue_but_not_start_a_track(self) -> None:
        rng = random.Random(211)
        config = AdaptiveDetectorConfig(
            window_events=300,
            min_events=80,
            hypotheses=900,
            update_interval_events=1,
            min_radius_px=4.0,
            max_radius_px=80.0,
            seed=211,
        )
        tracked = AdaptiveCircleDetector(config)
        cx, cy, radius = 64.0, 62.0, 25.0

        for index in range(300):
            angle = 2.0 * math.pi * index / 300.0
            tracked.push(
                FlowEvent(
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                    rng.randrange(4),
                    float(index),
                )
            )
        self.assertIsNotNone(tracked.detect(force=True))

        sparse_window = []
        for index in range(270):
            sparse_window.append(FlowEvent(cx, cy, index % 4, float(1000 + index)))
        for index in range(30):
            angle = 2.0 * math.pi * index / 30.0
            sparse_window.append(
                FlowEvent(
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                    rng.randrange(4),
                    float(1270 + index),
                )
            )
        tracked.extend(sparse_window)
        continued = tracked.detect(force=True)
        self.assertIsNotNone(continued)
        assert continued is not None
        self.assertLess(math.hypot(continued.cx - cx, continued.cy - cy), 1.0)
        self.assertLess(abs(continued.radius - radius), 0.8)

        fresh = AdaptiveCircleDetector(config)
        fresh.extend(sparse_window)
        self.assertIsNone(fresh.detect(force=True))


if __name__ == "__main__":
    unittest.main()
