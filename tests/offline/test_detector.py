from __future__ import annotations

import math
import random
import unittest

from circle_detection.detector import DetectorConfig, DirectionalCircleDetector, FlowEvent


def make_circle_events(
    cx: float,
    cy: float,
    radius: float,
    *,
    seed: int,
    circle_events: int = 220,
    outliers: int = 35,
) -> list[FlowEvent]:
    rng = random.Random(seed)
    events = []
    for i in range(circle_events):
        angle = rng.random() * 2.0 * math.pi
        x = cx + radius * math.cos(angle) + rng.gauss(0.0, 0.55)
        y = cy + radius * math.sin(angle) + rng.gauss(0.0, 0.55)
        code = int(round((angle % (2.0 * math.pi)) / (math.pi / 2.0))) % 4
        if rng.random() < 0.04:
            code = rng.randrange(4)
        events.append(FlowEvent(x, y, code, 1000.0 + i * 50.0))
    for i in range(outliers):
        events.append(
            FlowEvent(
                rng.uniform(0.0, 127.0),
                rng.uniform(0.0, 127.0),
                rng.randrange(4),
                1000.0 + rng.random() * circle_events * 50.0,
            )
        )
    events.sort(key=lambda event: event.t)
    return events


class DirectionalCircleDetectorTests(unittest.TestCase):
    def make_detector(self) -> DirectionalCircleDetector:
        return DirectionalCircleDetector(
            DetectorConfig(
                time_window=20_000.0,
                time_tau=20_000.0,
                max_events=512,
                radial_tolerance_px=3.0,
                min_radial_inlier_ratio=0.55,
            )
        )

    def test_arbitrary_centres_and_radii(self) -> None:
        cases = [
            (22.0, 24.0, 11.0),
            (64.0, 64.0, 29.0),
            (99.0, 35.0, 18.0),
            (37.0, 91.0, 25.0),
        ]
        for seed, (cx, cy, radius) in enumerate(cases, start=1):
            detector = self.make_detector()
            detector.extend(make_circle_events(cx, cy, radius, seed=seed))
            result = detector.detect()
            self.assertIsNotNone(result)
            assert result is not None
            self.assertLess(math.hypot(result.cx - cx, result.cy - cy), 1.6)
            self.assertLess(abs(result.radius - radius), 1.2)

    def test_rejects_unstructured_noise(self) -> None:
        rng = random.Random(42)
        detector = self.make_detector()
        detector.extend(
            FlowEvent(
                rng.uniform(0.0, 127.0),
                rng.uniform(0.0, 127.0),
                rng.randrange(4),
                float(i * 50),
            )
            for i in range(260)
        )
        self.assertIsNone(detector.detect())

    def test_four_unoriented_normal_axes(self) -> None:
        rng = random.Random(17)
        angles_deg = {0: 0.0, 1: 45.0, 2: 90.0, 3: 135.0}
        config = DetectorConfig(
            time_window=20_000.0,
            time_tau=20_000.0,
            max_events=512,
            direction_half_width_deg=22.5,
            radial_tolerance_px=3.0,
        )
        detector = DirectionalCircleDetector(config, angles_deg)
        cx, cy, radius = 78.0, 51.0, 24.0
        for i in range(260):
            angle = rng.random() * 2.0 * math.pi
            orientation = angle % math.pi
            code = min(
                angles_deg,
                key=lambda key: min(
                    abs(orientation - math.radians(angles_deg[key])),
                    math.pi - abs(orientation - math.radians(angles_deg[key])),
                ),
            )
            detector.push(
                FlowEvent(
                    cx + radius * math.cos(angle) + rng.gauss(0.0, 0.5),
                    cy + radius * math.sin(angle) + rng.gauss(0.0, 0.5),
                    code,
                    500.0 + i * 50.0,
                )
            )
        result = detector.detect()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(math.hypot(result.cx - cx, result.cy - cy), 1.0)
        self.assertLess(abs(result.radius - radius), 0.8)

    def test_rejects_crossed_straight_edges(self) -> None:
        rng = random.Random(71)
        detector = self.make_detector()
        events = []
        for i in range(130):
            events.append(FlowEvent(30.0, rng.uniform(10.0, 116.0), 0, float(i * 50)))
            events.append(FlowEvent(rng.uniform(10.0, 116.0), 70.0, 1, float(i * 50 + 1)))
        detector.extend(events)
        self.assertIsNone(detector.detect())

    def test_old_events_expire(self) -> None:
        detector = self.make_detector()
        detector.extend(make_circle_events(60.0, 55.0, 20.0, seed=9, outliers=0))
        self.assertIsNotNone(detector.detect())
        detector.push(FlowEvent(1.0, 1.0, 0, 100_000.0))
        self.assertIsNone(detector.detect())


if __name__ == "__main__":
    unittest.main()
