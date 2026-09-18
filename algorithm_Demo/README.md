# 硬件实时圆检测 Demo

当前硬件程序已经拆成两个独立模块：

- **`Demo_SNN.py`**：只保存 Speck2f SNN 网络、各层连接、Layer-4 输出、
  设备打开和 samnagui 可视化连接；
- **`Demo_algorithm.py`**：保存圆检测参数、过滤、终端状态、击球回放、
  运行日志和可执行主循环。

`Demo_SNN.py` 使用 `optical_flow_split_d_on_off_3_k_conv.py` 中的 split-D
设计作为网络结构参考：输入层采用 3x3 卷积，两级方向核采用无损裁剪后的
2x2 最小核。`Demo_algorithm.py` 使用项目上级
`circle_detection/AdaptiveCircleDetector` 的最新方法。

`Demo_SNN.py` 是配置模块，不单独启动事件循环。`Demo_algorithm.py`、
`Demo_record.py` 和 `DS_Demo.py` 都从它导入同一份网络配置，因此三条运行路径
不会各自维护一份 SNN。

## 数据流

```text
Speck2f Layer-4 事件
  -> split-D 前级：64x64 -> 63x63 -> 62x62
  -> 3x3、padding=2 的双斜率输出头：64x64x16
  -> 解码为 128x128 的 (x, y, direction, timestamp)
  -> 最近 230 个事件的自适应三点圆共识
  -> xiaoiron_confidence
  -> 可扩展具名过滤链
  -> 合格圆 (cx, cy, radius, confidence)
  -> 平滑半径峰值 / 上升后丢圆检测
  -> 击球前后事件慢速回放
```

Layer-4 的最终接口仍为 `64x64x16`。`0..7` 是第一组光流输出通道，
`8..15` 保留相同的光流方向和 2x2 子像素地址，但使用互补空间卷积核。
普通圆检测和 `DS_Demo.py` 都按原始光流方向合并对应的两组通道，再分别
累计 `D=y-x` 和 `S=x+y`。输出头自身的 weights 来自
`layer4_weights.py` 中选中的那套，阈值在 `Demo_SNN.py` 里手动输入，
详见下文"Layer-4 输出头的 weights 和阈值"。

检测器默认每 6 个输入事件更新一次，但每个事件都会按原始顺序进入窗口。
没有把检测器的缓存结果重复当作新圆输出。

## 调整参数

网络结构、层编号和监控开关在 `Demo_SNN.py` 中修改。

### Layer-4 输出头的 weights 和阈值

`layer4_weights.py` 里每套 weights 就是一个函数，显式写出 weights，并返回它
配套的 `kernel_size` / `padding` / `output_features` / `threshold_high` / `bias`。
`get_layer4_weights(index)` 用 if/elif（switch）按下标取用，共 13 套：

| 下标 | 名称 | 输出通道 | kernel | threshold_high | bias |
|---|---|---|---|---|---|
| `0` | `direct1` | 8 | 1x1 | 1 | 0 |
| `1` | `diag3` | 16 | 3x3 | 3 | -2 |
| `2` | `tangent3` | 16 | 3x3 | 3 | -2 |
| `3` | `diag5` | 16 | 5x5 | 4 | -3 |
| `4` | `diag5_long` | 16 | 5x5 | 3 | -2 |
| `5` | `diag3_a` | 8 | 3x3 | 3 | -2 |
| `6` | `diag3_b` | 8 | 3x3 | 3 | -2 |
| `7` | `tangent3_a` | 8 | 3x3 | 3 | -2 |
| `8` | `tangent3_b` | 8 | 3x3 | 3 | -2 |
| `9` | `diag5_a` | 8 | 5x5 | 4 | -3 |
| `10` | `diag5_b` | 8 | 5x5 | 4 | -3 |
| `11` | `diag5_long_a` | 8 | 5x5 | 3 | -2 |
| `12` | `diag5_long_b` | 8 | 5x5 | 3 | -2 |
| `13` | `diag7` | 16 | 7x7 | 5 | -4 |
| `14` | `diag7_a` | 8 | 7x7 | 5 | -4 |
| `15` | `diag7_b` | 8 | 7x7 | 5 | -4 |

`0` 是 1x1 核 `[[1]]` 的直通（按输出 `0..7` 的通道对应方式、不做空间卷积，
等价于旧代码里注释掉的那个 8 通道头，只是 padding 改为 1 以便把 62 补到
64）；`1..4` 与 `13` 是输出 16 通道的整头（前 8 个通道用主斜率核，后 8 个
通道换成互补斜率核）；`5..12` 与 `14..15` 是把每套整头按输出通道 `0..7` /
`8..15` 对半切开后的 8 通道版本（`_a` = 前 8 通道，`_b` = 后 8 通道，两者
拼起来就是原 16 通道）。每半各自的 D/S 家族分组不变。

新增的整套 weights 一律追加到已有下标之后，旧下标不会卷位。注意更大核占的
kernel SRAM 更多（`diag7` 整头 16x8x7x7 = 6272 字节，`diag3` 是 1152 字节），
能否放得下要看硬件。

`Demo_SNN.py` 不再写死 16 通道，而是用条目里的 `output_features`：

```python
# Demo_SNN.py
LAYER4_WEIGHT_INDEX = 0        # switch：0..15（0 = 1x1 直通）
LAYER4_THRESHOLD_LOW = -1      # 手动输入
```

`Demo_SNN.py` 只提供固定的输入接口（8 通道输入、62x62），输出通道数、
`kernel_size`、`padding`、`weights`、`threshold_high`、`bias` 全部来自
选中的条目。启动时会打印生效的下标、名称、输出通道数、核尺寸、padding、
weights 形状和阈值。其他入口（`Demo_algorithm.py`、`Demo_record.py`、
`DS_Demo.py`）不接 Layer-4 参数，自动使用同一份 weights。

注意：用 8 通道条目时，下游 128x128 解码和圆检测原本按 `64x64x16` 设计，
需要相应调整（两个 8 通道头各自连接后再合并）。

加新 weights：写一个 `weights_xxx()` 显式写出权重并返回上述字段，在
`get_layer4_weights()` 的 if/elif 里加一个分支，再把 `WEIGHT_COUNT` 加 1。
换卷积核尺寸时记得一起改 `padding`，让
`62 + 2 * padding - kernel_size + 1 == 64`。

圆检测常用参数集中在 `Demo_algorithm.py` 开头：

- `CIRCLE_FILTER_CONFIG`：输出置信度、半径和圆心范围；
- `SCORE_WINDOW_EVENTS`、`XIAOIRON_CONFIDENCE_CONFIG`：对应离线播放器中的
  `score N`、`alpha`、`lambda`、`N_min`、`W_min`、`K_min` 和 `sector W`；
- `CIRCLE_DETECTOR_CONFIG`：窗口、假设数、细化次数、更新间隔等算法参数；
- `HIT_REPLAY_CONFIG`：击球半径曲线、固定球杆点、回放窗口及播放速度；
- `TERMINAL_CONFIG`：终端刷新、日志和统计周期。

`accepted_output_interval_sec` 默认将终端圆输出合并到最多 10 Hz，避免稳定圆在
高事件率下产生数百次 `print/flush`。设为 `0` 可输出每一次通过的检测更新；
`accepted`、`terminal_outputs` 和 `coalesced` 计数会明确显示是否发生合并。

当前输出条件为：

```text
xiaoiron_confidence >= 0.30
35 <= radius <= 41
30 < cx < 90
40 < cy < 100
```

半径包含 35 和 41；圆心范围按开区间处理。

## 添加过滤条件

过滤实现位于 `circle_runtime.py`。新增一个返回 `FilterDecision` 的小函数，
再将 `CircleFilterRule("规则名", 函数)` 追加到 `CIRCLE_FILTER_RULES` 即可。
主循环、终端拒绝原因和每规则计数会自动接入，不需要再写一组嵌套 `if`。

## 击球时机和回放

`hit_replay.py` 使用事件环形缓冲区持续保存最近700毫秒的解码事件。合格几何圆的半径经过
EMA 平滑后，如果先上升再回落形成局部峰值，峰值时间即视为击球时刻；持续上升后连续丢圆
也可以触发，用于球在碰撞后快速离开画面的情况。800毫秒冷却时间避免同一次击球重复触发。

触发后继续采集到击球后200毫秒，再将击球前150毫秒至击球后200毫秒的事件交给独立
Matplotlib 进程以0.2倍速播放。实时采集和圆检测不会等待回放结束。画面中的绿色圆表示
击球时刻的球，红色叉号是固定球杆位置 `(68, 83)`，黄色虚线和归一化偏移表示球杆击在球
上的相对位置。

## 运行与测试

连接 Speck2f 硬件并准备好 `samna` / `samnagui` 后，在项目根目录运行：

```powershell
python algorithm_Demo\Demo_algorithm.py
```

硬件无关的解码、过滤边界和输出去重测试：

```powershell
python -m unittest discover -s tests/hardware -t . -p "test_*.py" -v
```

终端中的 `CANDIDATE` 是调参诊断；只有 `OUTPUT ACCEPT` 或
`[CIRCLE OUTPUT]` 行代表最终输出圆。性能行会显示事件率、更新率和检测耗时的
平均值 / P95 / 最大值。
