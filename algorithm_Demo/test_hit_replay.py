from dataclasses import replace
import unittest

from circle_detection import FlowEvent

from algorithm_Demo.hit_replay import (
    HitReplayConfig,
    HitTrigger,
    RadiusPeakHitDetector,
    RadiusSample,
    SlidingEventWindow,
    format_hit_trigger,
)


def observe_sequence(detector, radii, *, start_timestamp=0.0):
    trigger = None
    for index, radius in enumerate(radii, start=1):
        trigger = detector.observe_circle(
            update_number=index,
            source_event_count=index * 6,
            timestamp=start_timestamp + index,
            cx=60.0,
            cy=80.0,
            radius=float(radius),
            confidence=0.8,
        )
        if trigger is not None:
            return trigger
    return trigger


class RadiusPeakHitDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = HitReplayConfig(
            radius_ema_alpha=1.0,
            rise_window_samples=3,
            fall_confirm_samples=2,
            min_radius_rise_px=1.5,
            min_radius_fall_px=1.0,
            min_rising_fraction=2 / 3,
            min_falling_fraction=1.0,
            lost_confirm_updates=2,
            lost_min_radius_px=38.0,
            lost_min_radius_rise_px=2.0,
            cooldown_us=1_000.0,
        )

    def test_rise_then_fall_selects_the_radius_peak(self):
        detector = RadiusPeakHitDetector(self.config)
        trigger = observe_sequence(detector, [35.0, 36.0, 38.0, 40.0, 39.0, 38.0])

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.reason, "radius_peak")
        self.assertEqual(trigger.peak.radius, 40.0)
        self.assertEqual(trigger.peak.source_event_count, 24)
        self.assertEqual(trigger.radius_rise_px, 5.0)
        self.assertEqual(trigger.radius_fall_px, 2.0)

    def test_monotonic_or_small_jitter_does_not_form_a_peak(self):
        detector = RadiusPeakHitDetector(self.config)
        trigger = observe_sequence(detector, [35.0, 36.0, 37.0, 38.0, 39.0, 40.0])
        self.assertIsNone(trigger)

        detector = RadiusPeakHitDetector(self.config)
        trigger = observe_sequence(detector, [38.0, 38.1, 37.9, 38.2, 38.0, 37.8])
        self.assertIsNone(trigger)

    def test_rising_circle_followed_by_loss_can_trigger(self):
        detector = RadiusPeakHitDetector(self.config)
        self.assertIsNone(observe_sequence(detector, [35.0, 36.0, 37.0, 39.0]))
        self.assertIsNone(detector.observe_missing(timestamp=5.0))
        trigger = detector.observe_missing(timestamp=6.0)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.reason, "rising_then_lost")
        self.assertEqual(trigger.peak.radius, 39.0)

    def test_cooldown_blocks_a_second_nearby_peak(self):
        detector = RadiusPeakHitDetector(self.config)
        first = observe_sequence(
            detector,
            [35.0, 36.0, 38.0, 40.0, 39.0, 38.0],
            start_timestamp=0.0,
        )
        second = observe_sequence(
            detector,
            [35.0, 36.0, 38.0, 40.0, 39.0, 38.0],
            start_timestamp=100.0,
        )
        third = observe_sequence(
            detector,
            [35.0, 36.0, 38.0, 40.0, 39.0, 38.0],
            start_timestamp=2_000.0,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)
        self.assertEqual(third.hit_number, 2)


class SlidingEventWindowTests(unittest.TestCase):
    @staticmethod
    def make_trigger(timestamp=200.0):
        peak = RadiusSample(
            update_number=10,
            source_event_count=60,
            timestamp=timestamp,
            cx=60.0,
            cy=80.0,
            radius=40.0,
            smoothed_radius=40.0,
            confidence=0.8,
        )
        return HitTrigger(
            hit_number=1,
            reason="radius_peak",
            detected_timestamp=230.0,
            peak=peak,
            radius_rise_px=2.0,
            radius_fall_px=1.0,
            cue_x_px=68.0,
            cue_y_px=83.0,
        )

    def test_window_waits_for_post_hit_events_and_keeps_requested_range(self):
        config = replace(
            HitReplayConfig(),
            pre_hit_us=100.0,
            post_hit_us=150.0,
            buffer_duration_us=400.0,
        )
        window = SlidingEventWindow(config)
        for timestamp in range(0, 251, 10):
            window.push(FlowEvent(10.0, 20.0, 0, float(timestamp)))
        window.arm(self.make_trigger())

        self.assertEqual(window.pending_count, 1)
        self.assertEqual(window.pop_ready(), ())

        for timestamp in range(260, 351, 10):
            window.push(FlowEvent(10.0, 20.0, 0, float(timestamp)))
        clips = window.pop_ready()

        self.assertEqual(len(clips), 1)
        self.assertTrue(clips[0].complete)
        self.assertEqual(clips[0].events[0].t, 100.0)
        self.assertEqual(clips[0].events[-1].t, 350.0)

    def test_flush_marks_an_incomplete_post_hit_clip(self):
        config = replace(
            HitReplayConfig(),
            pre_hit_us=100.0,
            post_hit_us=150.0,
            buffer_duration_us=400.0,
        )
        window = SlidingEventWindow(config)
        for timestamp in range(100, 251, 10):
            window.push(FlowEvent(10.0, 20.0, 0, float(timestamp)))
        window.arm(self.make_trigger())

        clips = window.flush()
        self.assertEqual(len(clips), 1)
        self.assertFalse(clips[0].complete)
        self.assertEqual(clips[0].events[-1].t, 250.0)

    def test_trigger_reports_cue_position_relative_to_ball(self):
        trigger = self.make_trigger()
        self.assertEqual(trigger.cue_offset_x_px, 8.0)
        self.assertEqual(trigger.cue_offset_y_px, 3.0)
        self.assertAlmostEqual(trigger.cue_offset_x_radius, 0.2)
        self.assertAlmostEqual(trigger.cue_offset_y_radius, 0.075)
        line = format_hit_trigger(trigger)
        self.assertIn("ball=(60.000,80.000)", line)
        self.assertIn("cue=(68.000,83.000)", line)


if __name__ == "__main__":
    unittest.main()
