"""Layer-4 输出头的 weights：用 switch（下标）选。

Layer-4 输出头把前级 `62x62x8` 的 split-D 特征图变成 `64x64x16` 的硬件接口。
每套 weights 就是一个函数，显式写出 `weights` 以及它配套的
`kernel_size` / `padding` / `threshold_high` / `bias`。换卷积核尺寸时必须一起
改 `padding`，让 `62 + 2 * padding - kernel_size + 1 == 64`，否则输出不再是
64x64；`get_layer4_weights()` 会检查这些字段是否齐全、`weights.shape` 与
`kernel_size` 是否一致。

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
WEIGHT_COUNT = 4

_INPUT_FEATURES = 8


def weights_diag3():
    """下标 0（默认）：3x3 对角核，中心 2，两条对角 tap=1。等价于原来的手写权重。"""

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
        "threshold_high": 3,
        "bias": -2,
        "weights": weights,
    }


def weights_tangent3():
    """下标 1：3x3 配置，但交换两组斜率：主斜率用 /、互补斜率用 \。"""

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
        "threshold_high": 3,
        "bias": -2,
        "weights": weights,
    }


def get_layer4_weights(index=DEFAULT_WEIGHT_INDEX):
    """switch：按下标返回一套 weights（含 kernel_size 和 padding）。"""

    index = int(index)
    if index == 0:
        entry = weights_diag3()
    elif index == 1:
        entry = weights_tangent3()
    elif index == 2:
        entry = weights_diag5()
    elif index == 3:
        entry = weights_diag5_long()
    else:
        raise KeyError(
            f"未知的 Layer-4 weights 下标 {index}；"
            f"可选 0..{WEIGHT_COUNT - 1}（见 format_weights_table()）"
        )

    _check_weights(index, entry)
    return entry


def _check_weights(index, entry):
    """检查 weights 条目完整、weights 形状与 kernel_size 一致，避免上机后才发现写错。"""

    missing = {
        "name",
        "kernel_size",
        "padding",
        "threshold_high",
        "bias",
        "weights",
    } - set(entry)
    if missing:
        raise ValueError(
            f"weights[{index}] 缺少字段：{', '.join(sorted(missing))}"
        )

    kernel_size = entry["kernel_size"]
    weights = entry["weights"]
    expected = (LAYER4_FEATURE_COUNT, _INPUT_FEATURES, kernel_size, kernel_size)
    if weights.shape != expected:
        raise ValueError(
            f"weights[{index}] ({entry['name']}): weights.shape={weights.shape} "
            f"与 kernel_size={kernel_size} 不匹配，应为 {expected}"
        )


def format_weights_table():
    """列出全部 weights，便于挑下标。"""

    lines = [f"可选的 Layer-4 weights（下标 0..{WEIGHT_COUNT - 1}）：", ""]
    for index in range(WEIGHT_COUNT):
        entry = get_layer4_weights(index)
        kernel_size = entry["kernel_size"]
        taps = int(np.count_nonzero(entry["weights"][0]))
        lines.append(
            f"{index}  {entry['name']:<12} kernel={kernel_size}x{kernel_size} "
            f"padding={entry['padding']} taps_per_channel={taps} "
            f"threshold_high={entry['threshold_high']} bias={entry['bias']}"
        )
    return "\n".join(lines)


__all__ = (
    "DEFAULT_WEIGHT_INDEX",
    "WEIGHT_COUNT",
    "format_weights_table",
    "get_layer4_weights",
)
