# 硬件实时圆检测 Demo

`Demo.py` 使用 `optical_flow_split_d_on_off_3_k_conv.py` 的 split-D
前级网络：输入层采用 3x3 卷积，两级方向核采用无损裁剪后的 2x2 最小核。
实时识别部分使用项目上级 `circle_detection/AdaptiveCircleDetector` 的最新方法。

## 数据流

```text
Speck2f Layer-4 事件
  -> split-D 前级：64x64 -> 63x63 -> 62x62
  -> 5x5、padding=3 的双斜率输出头：64x64x16
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
累计 `D=y-x` 和 `S=x+y`。

检测器默认每 6 个输入事件更新一次，但每个事件都会按原始顺序进入窗口。
没有把检测器的缓存结果重复当作新圆输出。

## 调整参数

常用参数集中在 `Demo.py` 开头：

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
python algorithm_Demo\Demo.py
```

硬件无关的解码、过滤边界和输出去重测试：

```powershell
python -m unittest algorithm_Demo.test_circle_runtime -v
```

终端中的 `CANDIDATE` 是调参诊断；只有 `OUTPUT ACCEPT` 或
`[CIRCLE OUTPUT]` 行代表最终输出圆。性能行会显示事件率、更新率和检测耗时的
平均值 / P95 / 最大值。
