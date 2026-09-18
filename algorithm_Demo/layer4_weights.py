"""Layer-4 输出头的 weights：用 switch（下标）选。

Layer-4 输出头把前级 `62x62x8` 的 split-D 特征图变成 Layer-4 接口。每套
weights 就是一个函数，显式写出 `weights` 以及它配套的
`kernel_size` / `padding` / `output_features` / `threshold_high` / `bias`。

三组 weights：

* **1x1 直通**（下标 0）：核为 `[[1]]` 的单个 tap，按输出 0..7 的通道对应方式
  直通，不做空间卷积，输出 8 通道；
* **16 通道**（下标 1..4、13）：一个头输出 `64x64x16`，前 8 个通道用主斜率
  核，后 8 个通道保持同一光流方向、换成互补斜率核；
* **8 通道拆分**（下标 5..12、14..15）：把上面每套按输出通道 0..7 / 8..15
  对半切开，`_a` 是前 8 通道（主斜率核），`_b` 是后 8 通道（互补斜率核），
  配合两个 8 通道头使用。两半各自的 D/S 家族分组都不变，只是物理通道号
  从 0 开始。

新加的整套 weights 一律追加到已有下标之后，避免旧下标搬家。

`Demo_SNN.py` 用 `output_shape_feature=layer4_head["output_features"]` 决定
该层输出多少通道。注意：下游 128x128 解码和圆检测按 `64x64x16` 设计，
用 8 通道条目时要相应调整。

换卷积核尺寸时必须一起改 `padding`，让
`62 + 2 * padding - kernel_size + 1 == 64`，否则输出不再是 64x64；
`get_layer4_weights()` 会检查字段是否齐全、`weights.shape` 是否与
`output_features` / `kernel_size` 一致。

`threshold_low` 不在这个文件里，在 `Demo_SNN.py` 手动输入：

    LAYER4_WEIGHT_INDEX = 0        # switch：按下标换 weights
    LAYER4_THRESHOLD_LOW = -1      # 手动输入

加新 weights：写一个 `weights_xxx()`，在 `get_layer4_weights()` 的 if/elif 里
加一个分支，再把 `WEIGHT_COUNT` 加 1。
"""

from __future__ import annotations

import numpy as np

try:  # 作为包导入时
    from .layer4_layout import LAYER4_FEATURE_COUNT
except ImportError:  # 直接从 algorithm_Demo/ 运行或导入时
    from layer4_layout import LAYER4_FEATURE_COUNT


DEFAULT_WEIGHT_INDEX = 0

# switch 支持的下标个数（get_layer4_weights 的 if/elif 必须覆盖 0..WEIGHT_COUNT-1）。
WEIGHT_COUNT = 16

# 下标分组：0 是 1x1 直通；1..4、13 是 16 通道整头；5..12、14..15 是每套按
# 输出通道 0..7 / 8..15 拆开的 8 通道版本（_a = 前 8 通道，_b = 后 8 通道）。
DIRECT_WEIGHT_INDEX = 0
FULL_WEIGHT_INDICES = (1, 2, 3, 4, 13)
SPLIT_WEIGHT_INDICES = (5, 6, 7, 8, 9, 10, 11, 12, 14, 15)

_INPUT_FEATURES = 8

# 允许的输出通道数：16 通道整头，或 1x1 直通 / 对半切开后的 8 通道。
_HALF_FEATURES = LAYER4_FEATURE_COUNT // 2
_ALLOWED_OUTPUT_FEATURES = (LAYER4_FEATURE_COUNT, _HALF_FEATURES)


# 输出 0..7 的通道对应方式：输出特征 -> 输入特征。
# 右下、左上（0/3/4/7）接输入 0/1/2/3，左下、右上（1/2/5/6）接输入 4/5/6/7。
_ROUTING_0_TO_7 = {
    0: 0,
    3: 1,
    4: 2,
    7: 3,
    1: 4,
    2: 5,
    5: 6,
    6: 7,
}


def weights_direct1():
    """下标 0：1x1 核 [[1]]，按输出 0..7 的通道对应方式直通，输出 8 通道。

    单 tap、无空间卷积，相当于把前级特征直接透传，可用作对照。
    """

    weights = np.zeros((_HALF_FEATURES, _INPUT_FEATURES, 1, 1), dtype=np.int8)
    for output_feature, input_feature in _ROUTING_0_TO_7.items():
        weights[output_feature, input_feature, 0, 0] = 1

    return {
        "name": "direct1",
        "kernel_size": 1,
        "padding": 1,
        "output_features": _HALF_FEATURES,
        "threshold_high": 1,
        "bias": 0,
        "weights": weights,
    }


def weights_diag3():
    """下标 0（默认）：3x3 对角核，中心 2，两条对角 tap=1，输出 16 通道。"""

    kernel_main = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ], dtype=np.int8)

    kernel_anti = np.array([
        [0, 0, 1],
        [0, 2, 0],
        [1, 0, 0],
    ], dtype=np.int8)

    weights = np.zeros((LAYER4_FEATURE_COUNT, _INPUT_FEATURES, 3, 3), dtype=np.int8)

    # 右下、左上：空间方向都是 \
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 /
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    # 第二组保持光流方向不变，但交换空间斜率。
    # 右下、左上：空间方向改为 /
    weights[8, 0] = kernel_anti
    weights[11, 1] = kernel_anti
    weights[12, 2] = kernel_anti
    weights[15, 3] = kernel_anti

    # 左下、右上：空间方向改为 \
    weights[9, 4] = kernel_main
    weights[10, 5] = kernel_main
    weights[13, 6] = kernel_main
    weights[14, 7] = kernel_main

    return {
        "name": "diag3",
        "kernel_size": 3,
        "padding": 2,
        "output_features": LAYER4_FEATURE_COUNT,
        "threshold_high": 3,
        "bias": -2,
        "weights": weights,
    }


def weights_tangent3():
    """下标 1：3x3 配置，但交换两组斜率：主斜率用 /、互补斜率用 \，输出 16 通道。"""

    entry = weights_diag3()
    weights = entry["weights"]

    kernel_main = np.array([
        [0, 0, 1],
        [0, 2, 0],
        [1, 0, 0],
    ], dtype=np.int8)

    kernel_anti = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ], dtype=np.int8)

    # 右下、左上：空间方向都是 /
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 \
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    # 第二组交换空间斜率
    weights[8, 0] = kernel_anti
    weights[11, 1] = kernel_anti
    weights[12, 2] = kernel_anti
    weights[15, 3] = kernel_anti
    weights[9, 4] = kernel_main
    weights[10, 5] = kernel_main
    weights[13, 6] = kernel_main
    weights[14, 7] = kernel_main

    entry["name"] = "tangent3"
    return entry


def weights_diag5():
    """下标 2：5x5 对角核，中心 2，半径 1 和 2 的对角 tap 各为 1；padding 改为 3。"""

    kernel_main = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.int8)

    kernel_anti = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.int8)

    weights = np.zeros((LAYER4_FEATURE_COUNT, _INPUT_FEATURES, 5, 5), dtype=np.int8)

    # 右下、左上：空间方向都是 \
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 /
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    # 第二组保持光流方向不变，但交换空间斜率。
    weights[8, 0] = kernel_anti
    weights[11, 1] = kernel_anti
    weights[12, 2] = kernel_anti
    weights[15, 3] = kernel_anti
    weights[9, 4] = kernel_main
    weights[10, 5] = kernel_main
    weights[13, 6] = kernel_main
    weights[14, 7] = kernel_main

    return {
        "name": "diag5",
        "kernel_size": 5,
        "padding": 3,
        "output_features": LAYER4_FEATURE_COUNT,
        "threshold_high": 4,
        "bias": -3,
        "weights": weights,
    }


def weights_diag5_long():
    """下标 3：5x5 对角核，只保留半径 2 的远端 tap，得到更长基线的斜率证据。"""

    kernel_main = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.int8)

    kernel_anti = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.int8)

    weights = np.zeros((LAYER4_FEATURE_COUNT, _INPUT_FEATURES, 5, 5), dtype=np.int8)

    # 右下、左上：空间方向都是 \
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 /
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    # 第二组保持光流方向不变，但交换空间斜率。
    weights[8, 0] = kernel_anti
    weights[11, 1] = kernel_anti
    weights[12, 2] = kernel_anti
    weights[15, 3] = kernel_anti
    weights[9, 4] = kernel_main
    weights[10, 5] = kernel_main
    weights[13, 6] = kernel_main
    weights[14, 7] = kernel_main

    return {
        "name": "diag5_long",
        "kernel_size": 5,
        "padding": 3,
        "output_features": LAYER4_FEATURE_COUNT,
        "threshold_high": 3,
        "bias": -2,
        "weights": weights,
    }


def weights_diag7():
    """下标 13：7x7 对角核，中心 2，半径 1/2/3 的对角 tap 各为 1；padding 改为 4。"""

    kernel_main = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.int8)

    kernel_anti = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)

    weights = np.zeros((LAYER4_FEATURE_COUNT, _INPUT_FEATURES, 7, 7), dtype=np.int8)

    # 右下、左上：空间方向都是 \
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 /
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    # 第二组保持光流方向不变，但交换空间斜率。
    weights[8, 0] = kernel_anti
    weights[11, 1] = kernel_anti
    weights[12, 2] = kernel_anti
    weights[15, 3] = kernel_anti
    weights[9, 4] = kernel_main
    weights[10, 5] = kernel_main
    weights[13, 6] = kernel_main
    weights[14, 7] = kernel_main

    return {
        "name": "diag7",
        "kernel_size": 7,
        "padding": 4,
        "output_features": LAYER4_FEATURE_COUNT,
        "threshold_high": 5,
        "bias": -4,
        "weights": weights,
    }


# ── 8 通道版本：把上面每套按输出通道 0..7（_a）和 8..15（_b）切开 ──
def weights_diag3_a():
    """下标 4：diag3 的前 8 通道（主斜率核），输出 8 通道。"""

    entry = weights_diag3()
    entry["name"] = "diag3_a"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][:_HALF_FEATURES].copy()
    return entry


def weights_diag3_b():
    """下标 5：diag3 的后 8 通道（互补斜率核），输出 8 通道。"""

    entry = weights_diag3()
    entry["name"] = "diag3_b"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][_HALF_FEATURES:].copy()
    return entry


def weights_tangent3_a():
    """下标 6：tangent3 的前 8 通道，输出 8 通道。"""

    entry = weights_tangent3()
    entry["name"] = "tangent3_a"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][:_HALF_FEATURES].copy()
    return entry


def weights_tangent3_b():
    """下标 7：tangent3 的后 8 通道，输出 8 通道。"""

    entry = weights_tangent3()
    entry["name"] = "tangent3_b"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][_HALF_FEATURES:].copy()
    return entry


def weights_diag5_a():
    """下标 8：diag5 的前 8 通道（主斜率核），输出 8 通道。"""

    entry = weights_diag5()
    entry["name"] = "diag5_a"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][:_HALF_FEATURES].copy()
    return entry


def weights_diag5_b():
    """下标 9：diag5 的后 8 通道（互补斜率核），输出 8 通道。"""

    entry = weights_diag5()
    entry["name"] = "diag5_b"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][_HALF_FEATURES:].copy()
    return entry


def weights_diag5_long_a():
    """下标 10：diag5_long 的前 8 通道（主斜率核），输出 8 通道。"""

    entry = weights_diag5_long()
    entry["name"] = "diag5_long_a"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][:_HALF_FEATURES].copy()
    return entry


def weights_diag5_long_b():
    """下标 11：diag5_long 的后 8 通道（互补斜率核），输出 8 通道。"""

    entry = weights_diag5_long()
    entry["name"] = "diag5_long_b"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][_HALF_FEATURES:].copy()
    return entry


def weights_diag7_a():
    """下标 14：diag7 的前 8 通道（主斜率核），输出 8 通道。"""

    entry = weights_diag7()
    entry["name"] = "diag7_a"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][:_HALF_FEATURES].copy()
    return entry


def weights_diag7_b():
    """下标 15：diag7 的后 8 通道（互补斜率核），输出 8 通道。"""

    entry = weights_diag7()
    entry["name"] = "diag7_b"
    entry["output_features"] = _HALF_FEATURES
    entry["weights"] = entry["weights"][_HALF_FEATURES:].copy()
    return entry


def get_layer4_weights(index=DEFAULT_WEIGHT_INDEX):
    """switch：按下标返回一套 weights（含 kernel_size / padding / 输出通道数）。"""

    index = int(index)
    if index == 0:
        entry = weights_direct1()
    elif index == 1:
        entry = weights_diag3()
    elif index == 2:
        entry = weights_tangent3()
    elif index == 3:
        entry = weights_diag5()
    elif index == 4:
        entry = weights_diag5_long()
    elif index == 5:
        entry = weights_diag3_a()
    elif index == 6:
        entry = weights_diag3_b()
    elif index == 7:
        entry = weights_tangent3_a()
    elif index == 8:
        entry = weights_tangent3_b()
    elif index == 9:
        entry = weights_diag5_a()
    elif index == 10:
        entry = weights_diag5_b()
    elif index == 11:
        entry = weights_diag5_long_a()
    elif index == 12:
        entry = weights_diag5_long_b()
    elif index == 13:
        entry = weights_diag7()
    elif index == 14:
        entry = weights_diag7_a()
    elif index == 15:
        entry = weights_diag7_b()
    else:
        raise KeyError(
            f"未知的 Layer-4 weights 下标 {index}；"
            f"可选 0..{WEIGHT_COUNT - 1}（见 format_weights_table()）"
        )

    _check_weights(index, entry)
    return entry


def _check_weights(index, entry):
    """检查 weights 条目完整、形状与 output_features / kernel_size 一致。"""

    missing = {
        "name",
        "kernel_size",
        "padding",
        "output_features",
        "threshold_high",
        "bias",
        "weights",
    } - set(entry)
    if missing:
        raise ValueError(
            f"weights[{index}] 缺少字段：{', '.join(sorted(missing))}"
        )

    output_features = entry["output_features"]
    if output_features not in _ALLOWED_OUTPUT_FEATURES:
        raise ValueError(
            f"weights[{index}] ({entry['name']}): output_features={output_features} "
            f"只能是 {_ALLOWED_OUTPUT_FEATURES}"
        )

    kernel_size = entry["kernel_size"]
    weights = entry["weights"]
    expected = (output_features, _INPUT_FEATURES, kernel_size, kernel_size)
    if weights.shape != expected:
        raise ValueError(
            f"weights[{index}] ({entry['name']}): weights.shape={weights.shape} "
            f"与 output_features={output_features} / kernel_size={kernel_size} "
            f"不匹配，应为 {expected}"
        )


def format_weights_table():
    """列出全部 weights，便于挑下标。"""

    lines = [f"可选的 Layer-4 weights（下标 0..{WEIGHT_COUNT - 1}）：", ""]
    for index in range(WEIGHT_COUNT):
        entry = get_layer4_weights(index)
        kernel_size = entry["kernel_size"]
        taps = int(np.count_nonzero(entry["weights"][0]))
        lines.append(
            f"{index:<3}{entry['name']:<14} out={entry['output_features']:<3} "
            f"kernel={kernel_size}x{kernel_size} padding={entry['padding']} "
            f"taps_per_channel={taps} "
            f"threshold_high={entry['threshold_high']} bias={entry['bias']}"
        )
    return "\n".join(lines)


__all__ = (
    "DEFAULT_WEIGHT_INDEX",
    "DIRECT_WEIGHT_INDEX",
    "FULL_WEIGHT_INDICES",
    "SPLIT_WEIGHT_INDICES",
    "WEIGHT_COUNT",
    "format_weights_table",
    "get_layer4_weights",
)
