from __future__ import annotations

import math
from pathlib import Path
import random
from types import SimpleNamespace
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circle_detection import AdaptiveDetectorConfig, FlowEvent
from circle_runtime import (
    CircleDetectionPipeline,
    CircleFilterConfig,
    DetectionFilterChain,
    FEATURE_TO_DIRECTION,
    decode_layer4_event,
)
from layer4_layout import D_HOUGH_FEATURES, S_HOUGH_FEATURES


def fake_detection(**overrides):
    values = {
        "cx": 60.0,
        "cy": 70.0,
        "radius": 38.0,
        "xiaoiron_confidence": 0.60,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CircleFilterTests(unittest.TestCase):
    def setUp(self):
        self.chain = DetectionFilterChain(CircleFilterConfig())

    def failed_rules(self, detection):
        return {
            decision.rule_name
            for decision in self.chain.evaluate(detection).failures
        }

    def test_nominal_circle_passes_every_rule(self):
        report = self.chain.evaluate(fake_detection())
        self.assertTrue(report.passed)
        self.assertEqual(report.failures, ())

    def test_radius_interval_is_closed(self):
        self.assertTrue(self.chain.evaluate(fake_detection(radius=35.0)).passed)
        self.assertTrue(self.chain.evaluate(fake_detection(radius=41.0)).passed)
        self.assertEqual(
            self.failed_rules(fake_detection(radius=34.999)),
            {"radius"},
        )
        self.assertEqual(
            self.failed_rules(fake_detection(radius=41.001)),
            {"radius"},
        )

    def test_center_intervals_are_open(self):
        for center_x in (30.0, 90.0):
            with self.subTest(center_x=center_x):
                self.assertEqual(
                    self.failed_rules(fake_detection(cx=center_x)),
                    {"center_x"},
                )
        for center_y in (40.0, 100.0):
            with self.subTest(center_y=center_y):
                self.assertEqual(
                    self.failed_rules(fake_detection(cy=center_y)),
                    {"center_y"},
                )

    def test_non_finite_confidence_is_rejected(self):
        self.assertEqual(
            self.failed_rules(fake_detection(xiaoiron_confidence=math.nan)),
            {"confidence"},
        )

    def test_all_failure_causes_are_reported(self):
        report = self.chain.evaluate(
            fake_detection(
                cx=10.0,
                cy=110.0,
                radius=50.0,
                xiaoiron_confidence=0.10,
            )
        )
        self.assertFalse(report.passed)
        self.assertEqual(
            [failure.rule_name for failure in report.failures],
            ["confidence", "radius", "center_x", "center_y"],
        )


class Layer4DecodeTests(unittest.TestCase):
    def test_all_features_use_the_latest_four_direction_mapping(self):
        expected_xy_offsets = (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ) * 4
        self.assertEqual(len(FEATURE_TO_DIRECTION), 16)
        self.assertEqual(FEATURE_TO_DIRECTION[:8], FEATURE_TO_DIRECTION[8:])
        for feature, (offset_x, offset_y) in enumerate(expected_xy_offsets):
            with self.subTest(feature=feature):
                event = decode_layer4_event(11, 17, feature, 1234)
                self.assertEqual(event.x, 22 + offset_x)
                self.assertEqual(event.y, 34 + offset_y)
                self.assertEqual(event.c, FEATURE_TO_DIRECTION[feature])
                self.assertEqual(event.t, 1234)

    def test_ds_vote_families_partition_all_16_features(self):
        self.assertEqual(D_HOUGH_FEATURES | S_HOUGH_FEATURES, frozenset(range(16)))
        self.assertFalse(D_HOUGH_FEATURES & S_HOUGH_FEATURES)
        self.assertEqual(D_HOUGH_FEATURES, frozenset((0, 3, 4, 7, 8, 11, 12, 15)))


class FakeDetector:
    def __init__(self, detection):
        self.detection = detection
        self.events = []
        self.detect_calls = 0
        self.force_values = []

    def push(self, event):
        self.events.append(event)

    def detect(self, force=False):
        self.force_values.append(force)
        self.detect_calls += 1
        return self.detection

    def reset(self):
        self.events.clear()
        self.detect_calls = 0
        self.force_values.clear()


class PipelineTests(unittest.TestCase):
    def test_cached_detector_results_are_not_emitted_twice(self):
        detector = FakeDetector(fake_detection())
        pipeline = CircleDetectionPipeline(
            AdaptiveDetectorConfig(update_interval_events=3),
            CircleFilterConfig(),
            detector=detector,
        )

        updates = [
            pipeline.process_raw_event(
                10, 10, index % len(FEATURE_TO_DIRECTION), 1000 + index
            )
            for index in range(7)
        ]

        emitted = [update for update in updates if update is not None]
        self.assertEqual([update.source_event_count for update in emitted], [1, 4, 7])
        self.assertTrue(all(update.accepted for update in emitted))
        self.assertEqual(detector.detect_calls, 3)
        self.assertEqual(detector.force_values, [True, True, True])
        self.assertEqual(pipeline.stats.detector_updates, 3)
        self.assertEqual(pipeline.stats.accepted, 3)

    def test_invalid_event_is_counted_and_raises(self):
        pipeline = CircleDetectionPipeline(
            AdaptiveDetectorConfig(),
            CircleFilterConfig(),
            detector=FakeDetector(fake_detection()),
        )
        with self.assertRaises(ValueError):
            pipeline.process_raw_event(64, 0, 0, 0)
        self.assertEqual(pipeline.stats.invalid_events, 1)
        self.assertEqual(pipeline.stats.source_events, 0)


class ActualDetectorIntegrationTests(unittest.TestCase):
    def test_pipeline_recovers_a_circle_with_the_real_adaptive_detector(self):
        rng = random.Random(17)
        center_x, center_y, radius = 60.0, 70.0, 38.0
        events = []
        for index in range(240):
            angle = rng.random() * 2.0 * math.pi
            events.append(
                FlowEvent(
                    center_x + radius * math.cos(angle) + rng.gauss(0.0, 0.45),
                    center_y + radius * math.sin(angle) + rng.gauss(0.0, 0.45),
                    rng.randrange(4),
                    float(index),
                )
            )
        for index in range(60):
            events.append(
                FlowEvent(
                    rng.uniform(0.0, 127.0),
                    rng.uniform(0.0, 127.0),
                    rng.randrange(4),
                    float(1000 + index),
                )
            )
        rng.shuffle(events)

        pipeline = CircleDetectionPipeline(
            AdaptiveDetectorConfig(
                window_events=300,
                min_events=80,
                hypotheses=500,
                update_interval_events=299,
                min_radius_px=34.0,
                max_radius_px=42.0,
                seed=91,
            ),
            CircleFilterConfig(min_confidence=0.0),
        )
        updates = [pipeline.process_flow_event(event) for event in events]
        update = next(item for item in reversed(updates) if item is not None)

        self.assertIsNotNone(update.detection)
        detection = update.detection
        self.assertLess(math.hypot(detection.cx - center_x, detection.cy - center_y), 1.5)
        self.assertLess(abs(detection.radius - radius), 1.0)
        self.assertTrue(update.accepted)


if __name__ == "__main__":
    unittest.main()
