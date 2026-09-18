"""Layer-4 weights（``algorithm_Demo/layer4_weights.py``）的离线测试。

这些测试不导入 samna：weights 模块只依赖 NumPy，因此可以直接校验
diag3 与旧手写权重完全一致、每套 weights 的 padding 都能把 62 补到 64、
拆分条目拼回去等于整头，以及 switch 的错误处理。

测试里一律用 ``index_of(name)`` 找下标，避免下标调整后到处改数字。
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ALGORITHM_DIR = PROJECT_ROOT / "algorithm_Demo"
for path in (ALGORITHM_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import layer4_weights  # noqa: E402


LEGACY_MAIN = np.array(
    [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ],
    dtype=np.int8,
)
LEGACY_ANTI = np.array(
    [
        [0, 0, 1],
        [0, 2, 0],
        [1, 0, 0],
    ],
    dtype=np.int8,
)


def legacy_layer4_weights():
    """旧 Demo_SNN.py 里手写的 layer_4 weights，用作回归基准。"""

    weights = np.zeros((16, 8, 3, 3), dtype=np.int8)
    for feature, input_feature in ((0, 0), (3, 1), (4, 2), (7, 3)):
        weights[feature, input_feature] = LEGACY_MAIN
    for feature, input_feature in ((1, 4), (2, 5), (5, 6), (6, 7)):
        weights[feature, input_feature] = LEGACY_ANTI
    for feature, input_feature in ((8, 0), (11, 1), (12, 2), (15, 3)):
        weights[feature, input_feature] = LEGACY_ANTI
    for feature, input_feature in ((9, 4), (10, 5), (13, 6), (14, 7)):
        weights[feature, input_feature] = LEGACY_MAIN
    return weights


def legacy_direct_weights():
    """旧代码注释里的 1x1 直通权重（8 通道、0..7 通道对应）。"""

    weights = np.zeros((8, 8, 1, 1), dtype=np.int8)
    for output_feature, input_feature in (
        (0, 0), (3, 1), (4, 2), (7, 3), (1, 4), (2, 5), (5, 6), (6, 7)
    ):
        weights[output_feature, input_feature, 0, 0] = 1
    return weights


def index_of(name):
    for index in range(layer4_weights.WEIGHT_COUNT):
        if layer4_weights.get_layer4_weights(index)["name"] == name:
            return index
    raise AssertionError(f"weights 里没有 {name!r}")


class DirectWeightsTests(unittest.TestCase):
    def test_index_zero_is_the_direct_1x1_weights(self):
        self.assertEqual(layer4_weights.DEFAULT_WEIGHT_INDEX, 0)
        self.assertEqual(layer4_weights.DIRECT_WEIGHT_INDEX, 0)
        entry = layer4_weights.get_layer4_weights()
        self.assertEqual(entry["name"], "direct1")
        self.assertEqual(entry["kernel_size"], 1)
        self.assertEqual(entry["padding"], 1)
        self.assertEqual(entry["output_features"], 8)

    def test_direct_1x1_matches_the_old_commented_out_head(self):
        entry = layer4_weights.get_layer4_weights(0)
        self.assertEqual(entry["weights"].shape, (8, 8, 1, 1))
        self.assertTrue(
            np.array_equal(entry["weights"], legacy_direct_weights())
        )
        self.assertEqual(entry["threshold_high"], 1)
        self.assertEqual(entry["bias"], 0)

    def test_direct_1x1_keeps_the_output_aligned(self):
        entry = layer4_weights.get_layer4_weights(0)
        # 62 + 2 * padding - kernel_size + 1 == 64，且中心偏移与 3x3 版本一致。
        self.assertEqual(
            62 + 2 * entry["padding"] - entry["kernel_size"] + 1, 64
        )
        self.assertEqual(entry["padding"] - entry["kernel_size"] // 2, 1)


class DefaultWeightsTests(unittest.TestCase):
    def test_diag3_reproduces_the_legacy_weights(self):
        entry = layer4_weights.get_layer4_weights(index_of("diag3"))
        weights = entry["weights"]
        self.assertEqual(weights.shape, (16, 8, 3, 3))
        self.assertEqual(weights.dtype, np.int8)
        self.assertTrue(np.array_equal(weights, legacy_layer4_weights()))

    def test_diag3_carries_kernel_geometry_and_old_layer4_tunables(self):
        entry = layer4_weights.get_layer4_weights(index_of("diag3"))
        self.assertEqual(entry["kernel_size"], 3)
        self.assertEqual(entry["padding"], 2)
        self.assertEqual(entry["threshold_high"], 3)
        self.assertEqual(entry["bias"], -2)
        self.assertEqual(entry["output_features"], 16)

    def test_entry_carries_output_features_threshold_and_bias(self):
        # output_features / threshold_high / bias 跟 weights 一起返回；
        # threshold_low 在 Demo_SNN.py 里手动输入，不在条目里。
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                entry = layer4_weights.get_layer4_weights(index)
                self.assertEqual(
                    set(entry),
                    {
                        "name",
                        "kernel_size",
                        "padding",
                        "output_features",
                        "threshold_high",
                        "bias",
                        "weights",
                    },
                )
                self.assertNotIn("threshold_low", entry)
                self.assertIn(entry["output_features"], (16, 8))
                self.assertIsInstance(entry["threshold_high"], int)
                self.assertIsInstance(entry["bias"], int)

    def test_missing_entry_field_is_reported(self):
        bad = {
            "name": "bad",
            "kernel_size": 3,
            "padding": 2,
            "weights": np.zeros((16, 8, 3, 3), dtype=np.int8),
        }
        with self.assertRaises(ValueError) as caught:
            layer4_weights._check_weights(0, bad)
        self.assertIn("bias", str(caught.exception))
        self.assertIn("threshold_high", str(caught.exception))

    def test_unknown_index_lists_the_valid_range(self):
        with self.assertRaises(KeyError) as caught:
            layer4_weights.get_layer4_weights(layer4_weights.WEIGHT_COUNT)
        self.assertIn("0", str(caught.exception))

    def test_each_call_returns_fresh_weights(self):
        index = index_of("diag3")
        first = layer4_weights.get_layer4_weights(index)
        second = layer4_weights.get_layer4_weights(index)
        self.assertIsNot(first, second)
        self.assertIsNot(first["weights"], second["weights"])
        first["weights"][0, 0] = 0
        self.assertTrue(
            np.array_equal(second["weights"], legacy_layer4_weights())
        )


class EveryWeightsTests(unittest.TestCase):
    def test_switch_covers_every_index(self):
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                entry = layer4_weights.get_layer4_weights(index)
                self.assertIsInstance(entry["name"], str)

    def test_index_groups_cover_every_index(self):
        groups = (
            (layer4_weights.DIRECT_WEIGHT_INDEX,)
            + layer4_weights.FULL_WEIGHT_INDICES
            + layer4_weights.SPLIT_WEIGHT_INDICES
        )
        self.assertEqual(sorted(groups), list(range(layer4_weights.WEIGHT_COUNT)))

    def test_every_weights_keeps_the_64_by_64_interface(self):
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                entry = layer4_weights.get_layer4_weights(index)
                kernel_size = entry["kernel_size"]
                weights = entry["weights"]
                self.assertEqual(weights.shape[0], entry["output_features"])
                self.assertEqual(weights.shape[1], 8)
                self.assertEqual(weights.shape[2], kernel_size)
                self.assertEqual(weights.shape[3], kernel_size)
                # 62 + 2 * padding - kernel_size + 1 必须等于 64。
                self.assertEqual(
                    62 + 2 * entry["padding"] - kernel_size + 1,
                    64,
                )

    def test_every_weights_connects_one_input_feature_per_output_feature(self):
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                weights = layer4_weights.get_layer4_weights(index)["weights"]
                for feature in range(weights.shape[0]):
                    used = np.flatnonzero(np.any(weights[feature] != 0, axis=(1, 2)))
                    self.assertEqual(len(used), 1)

    def test_full_weights_swap_the_two_banks_spatially(self):
        for index in layer4_weights.FULL_WEIGHT_INDICES:
            with self.subTest(index=index):
                weights = layer4_weights.get_layer4_weights(index)["weights"]
                self.assertEqual(weights.shape[0], 16)
                for feature in range(8):
                    self.assertFalse(
                        np.array_equal(weights[feature], weights[feature + 8])
                    )

    def test_weights_shape_mismatch_is_reported(self):
        bad = {
            "name": "bad",
            "kernel_size": 5,
            "padding": 3,
            "output_features": 16,
            "threshold_high": 3,
            "bias": -2,
            "weights": np.zeros((16, 8, 3, 3), dtype=np.int8),
        }
        with self.assertRaises(ValueError):
            layer4_weights._check_weights(0, bad)

    def test_unexpected_output_features_is_reported(self):
        bad = {
            "name": "bad",
            "kernel_size": 3,
            "padding": 2,
            "output_features": 4,
            "threshold_high": 3,
            "bias": -2,
            "weights": np.zeros((4, 8, 3, 3), dtype=np.int8),
        }
        with self.assertRaises(ValueError):
            layer4_weights._check_weights(0, bad)

    def test_five_by_five_weights_are_stricter(self):
        # 5x5 的 tap 更多，阈值和抑制相应加强。
        entry = layer4_weights.get_layer4_weights(index_of("diag5"))
        self.assertEqual(entry["threshold_high"], 4)
        self.assertEqual(entry["bias"], -3)


class WeightDetailTests(unittest.TestCase):
    def test_diag5_uses_five_by_five_with_padding_three(self):
        entry = layer4_weights.get_layer4_weights(index_of("diag5"))
        self.assertEqual(entry["kernel_size"], 5)
        self.assertEqual(entry["padding"], 3)
        kernel = entry["weights"][0, 0]
        self.assertEqual(int(kernel[0, 0]), 1)
        self.assertEqual(int(kernel[1, 1]), 1)
        self.assertEqual(int(kernel[2, 2]), 2)
        self.assertEqual(int(kernel[4, 4]), 1)

    def test_diag5_long_only_keeps_the_outer_taps(self):
        entry = layer4_weights.get_layer4_weights(index_of("diag5_long"))
        kernel = entry["weights"][0, 0]
        self.assertEqual(int(kernel[0, 0]), 1)
        self.assertEqual(int(kernel[1, 1]), 0)
        self.assertEqual(int(kernel[2, 2]), 2)
        self.assertEqual(int(kernel[3, 3]), 0)
        self.assertEqual(int(kernel[4, 4]), 1)

    def test_tangent3_swaps_the_kernels(self):
        base = layer4_weights.get_layer4_weights(index_of("diag3"))["weights"]
        tangent = layer4_weights.get_layer4_weights(index_of("tangent3"))["weights"]
        # 主斜率族改用互补核，第二组再换回来。
        self.assertTrue(np.array_equal(tangent[0, 0], base[1, 4]))
        self.assertTrue(np.array_equal(tangent[1, 4], base[0, 0]))
        self.assertTrue(np.array_equal(tangent[8, 0], base[0, 0]))
        self.assertTrue(np.array_equal(tangent[9, 4], base[1, 4]))

    def test_diag7_uses_seven_by_seven_with_padding_four(self):
        entry = layer4_weights.get_layer4_weights(index_of("diag7"))
        self.assertEqual(entry["kernel_size"], 7)
        self.assertEqual(entry["padding"], 4)
        self.assertEqual(entry["output_features"], 16)
        self.assertEqual(entry["threshold_high"], 5)
        self.assertEqual(entry["bias"], -4)
        kernel = entry["weights"][0, 0]
        for radius in range(7):
            expected = 2 if radius == 3 else 1
            self.assertEqual(int(kernel[radius, radius]), expected)
        anti = entry["weights"][1, 4]
        self.assertEqual(int(anti[0, 6]), 1)
        self.assertEqual(int(anti[3, 3]), 2)
        self.assertEqual(int(anti[6, 0]), 1)

    def test_table_lists_every_index(self):
        table = layer4_weights.format_weights_table()
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                self.assertIn(str(index), table)
        for name in ("direct1", "diag3", "diag5_long_b"):
            self.assertIn(name, table)
        self.assertIn("threshold_high", table)
        self.assertIn("bias", table)


class SplitWeightsTests(unittest.TestCase):
    def test_total_count_is_three_times_the_full_weights_plus_direct(self):
        # 每套整头 × 3（整头 + _a + _b）+ 1 套 1x1 直通。
        self.assertEqual(
            layer4_weights.WEIGHT_COUNT,
            3 * len(layer4_weights.FULL_WEIGHT_INDICES) + 1,
        )
        self.assertEqual(len(layer4_weights.FULL_WEIGHT_INDICES), 5)

    def test_full_and_split_indices_are_listed_separately(self):
        self.assertEqual(layer4_weights.FULL_WEIGHT_INDICES, (1, 2, 3, 4, 13))
        self.assertEqual(
            layer4_weights.SPLIT_WEIGHT_INDICES,
            (5, 6, 7, 8, 9, 10, 11, 12, 14, 15),
        )

    def test_split_entries_output_eight_channels(self):
        for index in layer4_weights.SPLIT_WEIGHT_INDICES:
            with self.subTest(index=index):
                entry = layer4_weights.get_layer4_weights(index)
                self.assertEqual(entry["output_features"], 8)
                self.assertEqual(entry["weights"].shape[0], 8)

    def test_a_and_b_together_equal_the_full_weights(self):
        for full_index in layer4_weights.FULL_WEIGHT_INDICES:
            with self.subTest(index=full_index):
                full = layer4_weights.get_layer4_weights(full_index)
                first = layer4_weights.get_layer4_weights(
                    index_of(full["name"] + "_a")
                )
                second = layer4_weights.get_layer4_weights(
                    index_of(full["name"] + "_b")
                )
                self.assertTrue(
                    np.array_equal(first["weights"], full["weights"][:8])
                )
                self.assertTrue(
                    np.array_equal(second["weights"], full["weights"][8:])
                )
                self.assertTrue(
                    np.array_equal(
                        np.concatenate([first["weights"], second["weights"]]),
                        full["weights"],
                    )
                )

    def test_split_entries_keep_the_full_geometry_and_tunables(self):
        for full_index in layer4_weights.FULL_WEIGHT_INDICES:
            full = layer4_weights.get_layer4_weights(full_index)
            with self.subTest(index=full_index):
                for suffix in ("_a", "_b"):
                    entry = layer4_weights.get_layer4_weights(
                        index_of(full["name"] + suffix)
                    )
                    for field in (
                        "kernel_size",
                        "padding",
                        "threshold_high",
                        "bias",
                    ):
                        self.assertEqual(entry[field], full[field])

    def test_split_weights_are_copies(self):
        index = index_of("diag3_a")
        first = layer4_weights.get_layer4_weights(index)
        second = layer4_weights.get_layer4_weights(index)
        self.assertIsNot(first["weights"], second["weights"])
        first["weights"][0, 0] = 0
        full = layer4_weights.get_layer4_weights(index_of("diag3"))["weights"]
        self.assertTrue(np.array_equal(second["weights"], full[:8]))


if __name__ == "__main__":
    unittest.main()
