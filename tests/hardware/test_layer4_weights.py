"""Layer-4 weights（``algorithm_Demo/layer4_weights.py``）的离线测试。

这些测试不导入 samna：weights 模块只依赖 NumPy，因此可以直接校验下标 0 的
weights 与旧手写权重完全一致、每套 weights 的 padding 都能把 62 补到 64，
以及 switch 的错误处理。
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
from layer4_layout import LAYER4_FEATURE_COUNT  # noqa: E402


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

    weights = np.zeros((LAYER4_FEATURE_COUNT, 8, 3, 3), dtype=np.int8)
    for feature, input_feature in ((0, 0), (3, 1), (4, 2), (7, 3)):
        weights[feature, input_feature] = LEGACY_MAIN
    for feature, input_feature in ((1, 4), (2, 5), (5, 6), (6, 7)):
        weights[feature, input_feature] = LEGACY_ANTI
    for feature, input_feature in ((8, 0), (11, 1), (12, 2), (15, 3)):
        weights[feature, input_feature] = LEGACY_ANTI
    for feature, input_feature in ((9, 4), (10, 5), (13, 6), (14, 7)):
        weights[feature, input_feature] = LEGACY_MAIN
    return weights


class DefaultWeightsTests(unittest.TestCase):
    def test_index_zero_reproduces_the_legacy_weights(self):
        entry = layer4_weights.get_layer4_weights()
        weights = entry["weights"]
        self.assertEqual(weights.shape, (16, 8, 3, 3))
        self.assertEqual(weights.dtype, np.int8)
        self.assertTrue(np.array_equal(weights, legacy_layer4_weights()))

    def test_index_zero_is_the_default_and_carries_kernel_geometry(self):
        entry = layer4_weights.get_layer4_weights()
        self.assertEqual(layer4_weights.DEFAULT_WEIGHT_INDEX, 0)
        self.assertEqual(entry["name"], "diag3")
        self.assertEqual(entry["kernel_size"], 3)
        self.assertEqual(entry["padding"], 2)

    def test_entry_carries_threshold_high_and_bias(self):
        # threshold_high / bias 跟 weights 一起由 get_layer4_weights 返回；
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
                        "threshold_high",
                        "bias",
                        "weights",
                    },
                )
                self.assertNotIn("threshold_low", entry)
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
        first = layer4_weights.get_layer4_weights(0)
        second = layer4_weights.get_layer4_weights(0)
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

    def test_every_weights_keeps_the_64_by_64_by_16_interface(self):
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                entry = layer4_weights.get_layer4_weights(index)
                kernel_size = entry["kernel_size"]
                weights = entry["weights"]
                self.assertEqual(weights.shape[0], LAYER4_FEATURE_COUNT)
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
                for feature in range(LAYER4_FEATURE_COUNT):
                    used = np.flatnonzero(np.any(weights[feature] != 0, axis=(1, 2)))
                    self.assertEqual(len(used), 1)

    def test_every_weights_swaps_the_two_banks_spatially(self):
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                weights = layer4_weights.get_layer4_weights(index)["weights"]
                for feature in range(8):
                    self.assertFalse(
                        np.array_equal(weights[feature], weights[feature + 8])
                    )

    def test_default_threshold_and_bias_match_the_old_layer4_call(self):
        entry = layer4_weights.get_layer4_weights()
        self.assertEqual(entry["threshold_high"], 3)
        self.assertEqual(entry["bias"], -2)

    def test_five_by_five_weights_are_stricter(self):
        # 5x5 的 tap 更多，阈值和抑制相应加强。
        self.assertEqual(layer4_weights.get_layer4_weights(2)["threshold_high"], 4)
        self.assertEqual(layer4_weights.get_layer4_weights(2)["bias"], -3)

    def test_weights_shape_mismatch_is_reported(self):
        bad = {
            "name": "bad",
            "kernel_size": 5,
            "padding": 3,
            "threshold_high": 3,
            "bias": -2,
            "weights": np.zeros((16, 8, 3, 3), dtype=np.int8),
        }
        with self.assertRaises(ValueError):
            layer4_weights._check_weights(2, bad)


class WeightDetailTests(unittest.TestCase):
    def test_index_two_uses_five_by_five_with_padding_three(self):
        entry = layer4_weights.get_layer4_weights(2)
        self.assertEqual(entry["name"], "diag5")
        self.assertEqual(entry["kernel_size"], 5)
        self.assertEqual(entry["padding"], 3)
        kernel = entry["weights"][0, 0]
        self.assertEqual(int(kernel[0, 0]), 1)
        self.assertEqual(int(kernel[1, 1]), 1)
        self.assertEqual(int(kernel[2, 2]), 2)
        self.assertEqual(int(kernel[4, 4]), 1)

    def test_index_three_only_keeps_the_outer_taps(self):
        kernel = layer4_weights.get_layer4_weights(3)["weights"][0, 0]
        self.assertEqual(int(kernel[0, 0]), 1)
        self.assertEqual(int(kernel[1, 1]), 0)
        self.assertEqual(int(kernel[2, 2]), 2)
        self.assertEqual(int(kernel[3, 3]), 0)
        self.assertEqual(int(kernel[4, 4]), 1)

    def test_index_one_swaps_the_kernels(self):
        base = layer4_weights.get_layer4_weights(0)["weights"]
        tangent = layer4_weights.get_layer4_weights(1)["weights"]
        # 主斜率族改用互补核，第二组再换回来。
        self.assertTrue(np.array_equal(tangent[0, 0], base[1, 4]))
        self.assertTrue(np.array_equal(tangent[1, 4], base[0, 0]))
        self.assertTrue(np.array_equal(tangent[8, 0], base[0, 0]))
        self.assertTrue(np.array_equal(tangent[9, 4], base[1, 4]))

    def test_table_lists_every_index(self):
        table = layer4_weights.format_weights_table()
        for index in range(layer4_weights.WEIGHT_COUNT):
            with self.subTest(index=index):
                self.assertIn(str(index), table)
        self.assertIn("diag5_long", table)
        self.assertIn("threshold_high", table)
        self.assertIn("bias", table)


if __name__ == "__main__":
    unittest.main()
