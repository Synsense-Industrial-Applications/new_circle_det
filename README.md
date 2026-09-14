# 13点四区域光流的逐事件圆检测

本目录是圆检测项目的独立移植版，默认读取：

`layer4_20260727_155031_part0001.csv`

原有的 `event_stream_player.py` 和数据文件保持不变。新程序不依赖该播放器缺失的
`online_ring_scorer`，只复用了并测试了其中的正确解码规则。

项目提供两种使用方式：

- `visualize_circle_detection.py`：读取 CSV 的离线逐事件播放器，可实时调整置信度公式参数；
- `algorithm_Demo/Demo.py`：连接 Speck2f 的硬件实时检测程序，使用同一套最新自适应圆检测器。

## 项目结构

```text
new_circle_det/
├─ circle_detection/          # 圆候选、跟踪和小铁置信度算法
├─ algorithm_Demo/            # Speck2f 硬件实时 Demo
├─ visualize_circle_detection.py
├─ four_region_flow.py        # Layer-4 事件坐标及方向解码
├─ test_*.py                  # 算法与播放器测试
├─ layer4_*.csv               # 离线示例事件流
└─ XIAOIRON_CONFIDENCE.md     # 置信度公式说明
```

## 硬件实时 Demo

准备好 Speck2f、`samna` 和 `samnagui` 后，在项目根目录运行：

```powershell
python algorithm_Demo\Demo.py
```

硬件 Demo 的可调参数集中在 `algorithm_Demo/Demo.py` 开头：

- `CIRCLE_FILTER_CONFIG`：最终输出的置信度、半径和圆心范围；
- `SCORE_WINDOW_EVENTS` 与 `XIAOIRON_CONFIDENCE_CONFIG`：对应离线播放器中的
  `score N`、`alpha`、`lambda`、`N_min`、`W_min`、`K_min` 和 `sector W`；
- `CIRCLE_DETECTOR_CONFIG`：候选生成、细化、更新频率等检测参数；
- `HIT_REPLAY_CONFIG`：半径峰值、击球冷却、事件窗口和慢放速度；
- `TERMINAL_CONFIG`：终端状态和输出刷新频率。

当前最终输出条件为：

```text
xiaoiron_confidence >= 0.30
35 <= radius <= 41
30 < center_x < 90
40 < center_y < 100
```

输出过滤采用具名规则链。新增过滤条件时，实现一个 `CircleFilterRule` 并追加到
`CIRCLE_FILTER_RULES`，主循环、拒绝原因和统计信息会自动接入。

### 击球时机与事件回放

硬件 Demo 对符合半径和圆心范围的检测结果继续维护一条 EMA 平滑半径曲线。以下任一情况
会触发一次击球：

1. 半径在最近一段检测中明显上升，形成局部最大值后又确认回落；
2. 半径持续上升到较大值，随后连续多个检测周期丢失目标。

第二条用于碰撞后球快速离开或被遮挡、来不及形成完整下降沿的情况。时机分析使用独立的
`min_timing_confidence=0.15` 保持半径曲线连续；这不会降低最终圆输出的 `0.30` 门限。
触发后系统从环形缓冲区截取击球前150毫秒和后200毫秒的 Layer-4 事件，并在独立进程中以
0.2倍速播放，因此不会阻塞硬件事件采集。回放画面会标出：

- 绿色圆和十字：击球峰值时检测到的球及球心；
- 红色叉号：固定球杆位置 `(68, 83)`；
- 黄色虚线：球心到球杆点的偏移；
- 标题：球杆点相对球半径的归一化坐标，可用于判断实际击球位置。

终端中的 `[HIT DETECTED]` 显示触发原因、峰值时间、球位置、球杆位置、上升/回落幅度；
`[HIT REPLAY ...]` 显示回放片段的事件数量、完整性和排队状态。

## 解码规则

```text
x128 = 2*x64 + ((feature % 4) // 2)
y128 = 2*y64 +  (feature % 4) % 2
c    = [0, 1, 1, 0, 2, 3, 3, 2][feature]
```

四个 `c` 分别是 `0=↘`、`1=↙`、`2=↖`、`3=↗`。这里不能使用旧数据的
`c=feature//4`，否则会把四方向错误压成两类。

## 直接运行

双击 `run_circle_detection.bat`。播放器底部提供 `Slower`、`Play/Pause`、`Faster`
按钮和时间倍率滑块，可在0.01×、0.025×、0.05×、0.1×、0.25×、0.5×、1×、2×、
5×、10×、20×之间切换。
1×时严格按照CSV的时间戳播放；默认显示最近160毫秒，旧事件按时间戳指数变淡。

`Progress` 进度条可以拖到任意事件；拖动时自动暂停。`Min conf` 可实时调整
0到1之间的显示阈值。默认紫色圆使用新 `xiaoiron_confidence`；点击 `Score: xiaoiron`
可切回原始 `confidence`（青色圆）。红色严格DSCT对照仍使用自己的原分数。
检测圆只有在对应置信度不低于阈值时才显示，阈值不会隐藏评分历史和调试统计。
`Loop range` 双端滑块选择循环播放的起止事件，点击 `Loop: Off` 使其变为
`Loop: On` 后，播放器会按照区间内的原始时间戳反复播放。

事件密集时一个刷新帧会包含该时间段内的多个事件，事件稀疏时会按真实时间等待；预计算阶段
所有事件仍按原始顺序逐个进入两个检测器，不会跳过算法输入。
空格暂停/继续，右方向键单步一个事件，`R`重新开始，`+/-`调整速度，`S`切换扇区编号和辅助线。

程序首次启动会先顺序计算完整事件流并生成同目录 `.circle_cache_v5.npz` 缓存。播放、
调阈值和拖进度条只读取缓存，不再现场运行检测器。数据或检测参数变化后缓存会自动失效；
也可用 `--rebuild-cache` 强制重建。

命令行：

```powershell
python visualize_circle_detection.py
python visualize_circle_detection.py --playback-speed 2
python visualize_circle_detection.py --trail-ms 200 --fade-tau-ms 60
python visualize_circle_detection.py --confidence-threshold 0.40
python visualize_circle_detection.py --rebuild-cache
python visualize_circle_detection.py --analyze --analysis-stride 50 --no-show
python visualize_circle_detection.py --save-preview circle_detection_preview.png --no-show
python visualize_circle_detection.py --save-gif circle_detection_event_stream.gif --gif-events 210 --no-show
```

若环境缺少依赖：

```powershell
python -m pip install -r requirements.txt
```

## 文件说明

- `four_region_flow.py`：与参考播放器一致的64×64×8到128×128四区域解码。
- `circle_detection/adaptive_detector.py`：固定事件数、三点圆假设和连续跟踪检测器。
- `circle_detection/xiaoiron_confidence.py`：独立新置信度函数与所有可调参数。
- `XIAOIRON_CONFIDENCE.md`：新公式、扇区规则、调参方法和验证说明。
- `circle_detection_step_by_step.ipynb`：逐事件主循环、输入输出调试、弹窗播放器、任意事件扇区检查。
- `circle_detection/detector.py`：原始方向约束DSCT，用作对照。
- `visualize_circle_detection.py`：按时间戳播放、时间渐隐和两种检测器对比。
- `test_*.py`：解码与检测器单元测试。

Adaptive检测器默认每6个事件更新一次几何候选，但所有源事件都会按原始顺序进入检测器。
`feature`方向只作为弱验证信号，圆的位置和半径主要由事件几何一致性决定。

每个角度扇区还必须累积足够的时间衰减事件权重，单个偶然事件不能算作有效覆盖。这会
排除由画面上边沿和左边沿组成的 L 形结构被拟合成超大圆的常见误识别。

检测结果由连续轨迹门控。相邻更新只有在圆心和半径变化均位于允许范围内才会平滑更新；
圆心单次变化限制为7像素，半径单次变化限制为4.5像素。远处或突然变大的候选不会单帧
替换当前圆，必须在旧轨迹失效后连续3次出现在相近位置和
半径才会建立新轨迹。这样会让疑似跳变短暂不显示，而不会把错误大圆画出来。

已经确认的轨迹允许在一次检测更新中使用较弱但几何一致的事件证据，从而跨过约6个事件
的短暂稀疏区。该宽松条件不能建立新圆，也不会修改严格轨迹状态；可能发生轨迹切换前
仍强制保留至少一次空白更新，因此不会为了补漏而重新引入跳变。

四方向一致性和最多10个候选圆的 Gauss-Newton 细化均使用 NumPy 批量向量运算，并缓存
固定事件窗口的衰减权重。优化不改变候选、置信度或跟踪结果；真实数据前20,000个事件
耗时由10.17秒降到5.91秒，检测数组与优化前逐项一致（最大数值差小于 `1e-5`）。

## 小铁新置信度（2026-09-08）

`xiaoiron_confidence = clip(full_confidence × 遮挡奖励 × 方向一致性系数, 0, 1)`。
它以完整圆几何置信度为基础。
默认遮挡奖励最高乘1.51；空缺奖励检查10/11/12/13，方向项使用
`R=min(q1*p1,q2*p2,q5*p5,q6*p6)`，按
`exp(-0.85*(1-R))` 扣分，因此事件不足和方向混杂都会降低置信度。默认
`N_min=5`、`W_min=1.75`、`K_min=6`、`sector W=1.0`。详见 [计算说明](XIAOIRON_CONFIDENCE.md)。

播放器上方左侧显示原始分数和每个系数，中间显示16扇区四方向堆叠柱形图；信息不覆盖事件图。
窗口右侧固定显示完整公式、当前事件的数值代入过程，以及 `alpha`、`lambda`、`N_min`、
`W_min`、`K_min`、`sector W` 六个实时参数滑块。拖动后会立即更新当前圆、显示过滤和整条
`xiaoiron` 历史曲线；`Reset formula params` 可恢复启动播放器时的参数。
下方历史同时显示 `full base` 和紫色 `xiaoiron`。
评分窗口是最近300个事件，默认每6个事件重算；显示拖尾则按时间戳，二者不是同一个窗口。

实时滑块只重算缓存的评分证据，不重跑圆几何生成和跟踪；15.5万个事件的整条评分曲线约30毫秒
即可更新。要修改公式结构，请编辑 `visualize_circle_detection.py` 中唯一的
`recompute_xiaoiron_series()` 公式区块，播放器其余代码无需修改。

Jupyter请重启内核并从头运行：参数cell中的 `XIAOIRON_CONFIG` 会作为播放器滑块的初始值，
事件循环后单独执行播放器cell。
最后的检查cell可设 `INSPECT_EVENT`（从1开始）输出扇区计数、四方向权重、纯度和是否完整位于画面内。
旧内存结果的方向证据只有2个扇区，不能只运行最后的播放器cell。脚本会重新生成v6缓存。

评分与显示过滤不修改完整圆候选生成、接受门限和跟踪。原检测器已经拒绝的
候选不会因新分数而复活；原 `confidence` 字段保留，新增 `detection.xiaoiron_confidence`
和 `detection.xiaoiron` 诊断对象供代码调用。
