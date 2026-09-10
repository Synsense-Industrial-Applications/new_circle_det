# 硬件实时圆检测 Demo

`Demo.py` 保留原 Speck2f/CNN 配置，实时识别部分已替换为项目上级
`circle_detection/AdaptiveCircleDetector` 的最新方法。

## 数据流

```text
Speck2f Layer-4 事件
  -> 64x64x8 解码为 128x128 的 (x, y, direction, timestamp)
  -> 最近 230 个事件的自适应三点圆共识
  -> xiaoiron_confidence
  -> 可扩展具名过滤链
  -> 合格圆 (cx, cy, radius, confidence)
```

检测器默认每 6 个输入事件更新一次，但每个事件都会按原始顺序进入窗口。
没有把检测器的缓存结果重复当作新圆输出。

## 调整参数

常用参数集中在 `Demo.py` 开头：

- `CIRCLE_FILTER_CONFIG`：输出置信度、半径和圆心范围；
- `SCORE_WINDOW_EVENTS`、`XIAOIRON_CONFIDENCE_CONFIG`：对应离线播放器中的
  `score N`、`alpha`、`lambda`、`N_min`、`W_min`、`K_min` 和 `sector W`；
- `CIRCLE_DETECTOR_CONFIG`：窗口、假设数、细化次数、更新间隔等算法参数；
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
