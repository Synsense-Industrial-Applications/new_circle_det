from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from circle_detection.xiaoiron_confidence import (
    XiaoironConfig, calculate_xiaoiron_confidence, sector_ids,
)


def score_fixture(empty=(), mixed=(), sparse=False, cy=64, config=XiaoironConfig()):
    sectors = np.repeat([s for s in range(16) if s not in empty], 20)
    angles = -np.pi + (sectors+0.5)*np.pi/8
    x, y = 64+25*np.cos(angles), cy+25*np.sin(angles)
    codes = np.zeros(len(sectors), dtype=int)
    codes[sectors == 6] = 3
    for s in mixed:
        codes[sectors == s] = np.arange((sectors == s).sum()) % 4
    if sparse:
        keep = (sectors != 2) & (sectors != 6)
        x, y, codes = x[keep], y[keep], codes[keep]
    return calculate_xiaoiron_confidence(
        x, y, codes, np.ones(len(x)), cx=64, cy=cy, radius=25,
        radial_tolerance_px=2.5, full_confidence=0.4, config=config)


class XiaoironConfidenceTests(unittest.TestCase):
    def test_default_sector_groups_match_current_rule(self):
        config = XiaoironConfig()
        self.assertEqual(config.occluded_sectors, (10, 11, 12, 13))
        self.assertEqual(config.direction_sectors, (1, 2, 5, 6))
        self.assertEqual(config.occlusion_gain, 0.51)
        self.assertEqual(config.direction_penalty_lambda, 0.85)
        self.assertEqual(config.min_direction_events, 5)
        self.assertEqual(config.min_direction_weight, 1.75)
        self.assertEqual(config.min_other_sectors, 6)
        self.assertEqual(config.min_sector_weight, 1.0)

    def test_sector_mapping_matches_existing_zero_based_convention(self):
        a = -np.pi+(np.arange(16)+0.5)*np.pi/8
        np.testing.assert_array_equal(sector_ids(np.cos(a), np.sin(a)), np.arange(16))

    def test_one_empty_bottom_sector_increases_full_score(self):
        baseline = score_fixture()
        empty = score_fixture(empty=(12,))
        self.assertAlmostEqual(baseline.xiaoiron_confidence, 0.4)
        self.assertAlmostEqual(empty.xiaoiron_confidence, 0.604)
        self.assertEqual(empty.empty_visible_count, 1)
        self.assertAlmostEqual(score_fixture(empty=(10,)).xiaoiron_confidence, 0.604)
        # Empty more sectors cannot compound the bonus without limit.
        self.assertAlmostEqual(score_fixture(empty=(10,11,12,13)).xiaoiron_confidence, 0.604)

    def test_either_mixed_shoulder_penalizes_score(self):
        for s in (1, 2, 5, 6):
            mixed = score_fixture(empty=(12,), mixed=(s,))
            self.assertLess(mixed.xiaoiron_confidence, 0.4)
            self.assertAlmostEqual(mixed.sector_purity[s], 0.25)
            self.assertAlmostEqual(mixed.direction_evidence, 0.25)
            self.assertAlmostEqual(mixed.direction_factor, np.exp(-0.85*0.75))

    def test_low_count_is_penalized_even_when_direction_is_pure(self):
        sectors = np.repeat(np.arange(16), 20)
        keep = np.ones(len(sectors), dtype=bool)
        shoulder_1 = np.flatnonzero(sectors == 1)
        keep[shoulder_1[4:]] = False
        sectors = sectors[keep]
        a = -np.pi+(sectors+0.5)*np.pi/8
        result = calculate_xiaoiron_confidence(
            64+25*np.cos(a), 64+25*np.sin(a), np.zeros(len(a)), np.ones(len(a)),
            cx=64, cy=64, radius=25, radial_tolerance_px=2.5, full_confidence=0.4)
        self.assertAlmostEqual(result.side_sufficiency[0], 0.8)
        self.assertAlmostEqual(result.sector_purity[1], 1.0)
        self.assertAlmostEqual(result.side_direction_evidence[0], 0.8)
        self.assertAlmostEqual(result.direction_factor, np.exp(-0.85*0.2))

    def test_tiny_weight_is_not_a_completely_empty_sector(self):
        sectors = np.repeat(np.arange(16), 20)
        a = -np.pi+(sectors+0.5)*np.pi/8
        weights = np.ones(len(a))
        weights[(sectors >= 11) & (sectors <= 13)] = 1e-12
        result = calculate_xiaoiron_confidence(
            64+25*np.cos(a), 64+25*np.sin(a), np.zeros(len(a)), weights,
            cx=64, cy=64, radius=25, radial_tolerance_px=2.5, full_confidence=0.4)
        self.assertEqual(result.empty_visible_count, 0)
        self.assertEqual(result.occlusion_factor, 1)

    def test_non_ring_events_do_not_fill_empty_sectors(self):
        s = np.arange(16)
        a = -np.pi+(s+0.5)*np.pi/8
        radius = np.where((s >= 10) & (s <= 13), 10, 25)
        result = calculate_xiaoiron_confidence(
            64+radius*np.cos(a), 64+radius*np.sin(a), s%4, np.ones(16),
            cx=64, cy=64, radius=25, radial_tolerance_px=2.5, full_confidence=0.4)
        self.assertEqual(result.empty_visible_count, 4)
        self.assertEqual(result.sector_counts.sum(), 12)

    def test_invalid_configuration_and_input_fail_clearly(self):
        for changes in (dict(occlusion_gain=np.nan),
                        dict(direction_penalty_lambda=np.nan),
                        dict(direction_sectors=(1, 2.0, 5, 6)),
                        dict(direction_sectors=(2,)),
                        dict(width=0), dict(min_direction_events=0)):
            with self.assertRaises(ValueError):
                replace(XiaoironConfig(), **changes)
        with self.assertRaisesRegex(ValueError, "direction codes"):
            calculate_xiaoiron_confidence(
                [64], [39], [np.nan], [1], cx=64, cy=64, radius=25,
                radial_tolerance_px=2.5, full_confidence=0.4)

    def test_absent_shoulders_and_offscreen_arcs_do_not_reward(self):
        sparse = score_fixture(empty=(12,), sparse=True)
        self.assertEqual(sparse.occlusion_factor, 1)
        self.assertTrue(np.isnan(sparse.sector_purity[2]))
        self.assertEqual(sparse.direction_evidence, 0)
        self.assertAlmostEqual(sparse.direction_factor, np.exp(-0.85))
        self.assertAlmostEqual(sparse.xiaoiron_confidence, 0.4*np.exp(-0.85))
        clipped = score_fixture(empty=(11,12,13), cy=110)
        self.assertEqual(clipped.empty_visible_count, 0)
        self.assertEqual(clipped.occlusion_factor, 1)

    def test_zero_gains_disable_both_new_factors(self):
        neutral = score_fixture(empty=(12,), mixed=(2,),
                                config=replace(XiaoironConfig(), occlusion_gain=0,
                                               direction_penalty_lambda=0))
        self.assertEqual(neutral.xiaoiron_confidence, 0.4)

    def test_output_cache_and_display_threshold(self):
        from offline_tools.four_region_flow import FlowData
        from offline_tools.visualize_circle_detection import (
            to_flow_events, make_configs, precompute_detections,
            save_precomputed_cache, load_precomputed_cache, DSCTEventPlayer,
        )
        from circle_detection.adaptive_detector import AdaptiveCircleDetector
        a = np.linspace(0, 2*np.pi, 300, endpoint=False)
        data = FlowData(
            Path("test.csv"), 64+25*np.cos(a), 64+25*np.sin(a),
            np.arange(300)%4, np.arange(300)*100, np.arange(300)*0.0001,
            np.zeros(300), np.zeros(300))
        events = to_flow_events(data)
        strict, adaptive = make_configs()
        result = precompute_detections(events, strict, adaptive, show_progress=False)
        finite = np.flatnonzero(np.isfinite(result.adaptive.xiaoiron_confidence))
        self.assertGreater(len(finite), 0)
        # Persist diagnostics, not just the scalar confidence.
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)/"cache.npz"
            save_precomputed_cache(cache, "test", result)
            loaded = load_precomputed_cache(cache, "test", len(events))
            np.testing.assert_equal(loaded.adaptive.direction_weights, result.adaptive.direction_weights)
            np.testing.assert_equal(loaded.adaptive.side_sufficiency, result.adaptive.side_sufficiency)
            np.testing.assert_equal(loaded.adaptive.side_direction_evidence,
                                    result.adaptive.side_direction_evidence)
            self.assertEqual(loaded.adaptive.side_sufficiency.shape[1], 4)
        player = DSCTEventPlayer(data, events, result, trail_ms=160, fade_tau_ms=50,
                                 interval_ms=30, playback_speed=0.01,
                                 confidence_threshold=0.99, history_events=100, enable_timer=False)
        try:
            player.seek_for_preview(int(finite[-1]))
            self.assertIsNone(player.adaptive_detection)
            self.assertIn("hidden by threshold", player.info_text.get_text())
            self.assertTrue(np.isfinite(player.xiaoiron_conf_line.get_ydata()).any())
            player.xiaoiron_sliders["direction_penalty_lambda"].set_val(0.0)
            self.assertEqual(player.xiaoiron_config.direction_penalty_lambda, 0.0)
            valid_score = np.isfinite(result.adaptive.full_confidence)
            np.testing.assert_allclose(
                result.adaptive.direction_factor[valid_score], 1.0
            )
            np.testing.assert_allclose(
                result.adaptive.xiaoiron_confidence[valid_score],
                np.clip(
                    result.adaptive.full_confidence[valid_score]
                    * result.adaptive.occlusion_factor[valid_score],
                    0.0,
                    1.0,
                ),
                rtol=1e-6,
            )
            self.assertIn("CURRENT EVENT", player.formula_current_text.get_text())
            player._reset_xiaoiron_parameters()
            self.assertEqual(player.xiaoiron_config.direction_penalty_lambda, 0.85)
            player.threshold_slider.set_val(0)
            self.assertIsNotNone(player.adaptive_detection)
            player._toggle_score_metric()
            self.assertEqual(player.confidence_metric, "confidence")
        finally:
            plt.close(player.figure)


if __name__ == "__main__":
    unittest.main()
