# xiaoiron_confidence：球杆相机的可解释评分

这是基于现有 full 几何置信度的启发式评分，不是已经标定的概率，也不能仅凭缺失事件判断发生击打。
本次保留原圆检测和跟踪，只增加独立分数及播放器过滤，便于比较规则本身的影响。

## 1. 输入与扇区

输入：当前检测窗口的事件 `(x,y,c,t)`、窗口衰减权重、已经接受并平滑的圆 `(cx,cy,r)`，以及该圆的 full 基础分。
新规则只统计满足 `abs(hypot(x-cx,y-cy)-r) <= radial_tolerance_px` 的圆周带事件，默认厚度为半径两侧各2.5像素；圆内部事件不参加扇区计数。

编号严格沿用项目 `圆弧.txt` 的 **0–15编号**。图像坐标y向下：

```python
sector = floor((atan2(y-cy, x-cx) + pi) / (2*pi) * 16) % 16
```

1、2覆盖左上肩部，5、6覆盖右上肩部，10、11、12、13覆盖下方易遮挡区域。播放器可显示实际编号，按S开关辅助线。
默认统计窗口为最近300个事件，每6个事件更新一次；权重 `exp(-事件年龄 / 120)`。
这里沿用的是事件数衰减，不是毫秒衰减。播放器仍按CSV时间戳播放和渐隐；改变播放速度或拖尾时长不改变检测分数。

令 `n_s` 为扇区s的原始圆周事件数，`H[s,c]` 为其方向c的衰减权重总和，`W_s=sum_c H[s,c]`。

## 2. 基础分 full

对当前圆应用已有 full 表达式：

```text
C_full = clip(ρ × min(1,K/8) × min(1,Q/3)
              × exp(-MAD/2.5) × (0.95 + 0.10 A), 0, 1)
```

ρ是圆周内点权重比例，K是有效角度扇区数，Q是覆盖象限数，MAD是原实现的径向误差统计量，A是原四方向弱一致性指标。
8、3、2.5随检测器对应参数变化，上式列出默认值。
`full base` 是检测器通过完整圆约束后得到的基础几何分数。

## 3. 遮挡奖励：缺失必须有上下文证据

```text
E = 1，当10/11/12/13中至少一个扇区 n_s == 0，且该扇区圆周带完整位于传感器内
E = 0，其余情况
q_s = min(1, n_s / 5, W_s / 1.75)，s属于{1,2,5,6}
B = min(1, 10/11/12/13以外有效扇区数 / 6)
G = min(q_1, q_2, q_5, q_6, B)
O = 1 + 0.51 × E × G
```

有效扇区要求权重至少1。完全无事件用原始计数判断，不把“衰减后很小”当作零。
四个肩部扇区证据都充足且其他圆弧有覆盖时，空缺触发最高乘1.51的奖励；缺失多个下方扇区不会叠加奖励。
这样避免整个窗口稀疏就因空扇区增分。落在画面外的空扇区不能作为球杆遮挡证据。
四个方向扇区中任一扇区证据不足时奖励连续减弱；任一扇区完全无事件时奖励关闭。

## 4. 方向扣分：把事件充分程度与方向纯度合并

```text
p_s = max_c H[s,c] / W_s          # 方向纯度，无事件则未知
r_s = q_s × p_s                    # 单扇区有效同向光流证据
R_1,2,5,6 = min(r_1, r_2, r_5, r_6)
D = exp(-0.85 × (1-R_1,2,5,6))
```

`q`回答“事件够不够”，`p`回答“方向纯不纯”，乘积`r=q×p`要求二者同时成立。
四个方向扇区取最小值，因此任一扇区证据不足都会降低整体方向系数，其他扇区不能补偿。
某扇区没有事件时其`p`显示为n/a，并直接定义`r=0`，得到最大惩罚`exp(-0.85)≈0.427`。
该式分别看1、2、5、6内部的主方向占比，不要求四个扇区的主方向代码相同。

## 5. 最终分数与例子

```text
xiaoiron_confidence = clip(
    C_full × (1 + alpha × E × G)
           × exp(-lambda × (1 - min(q_1×p_1, q_2×p_2, q_5×p_5, q_6×p_6))),
    0, 1
)
```

默认 `alpha=0.51`、`lambda=0.85`，也就是
`xiaoiron_confidence = clip(C_full × O_occlusion × D_1,2,5,6, 0, 1)`。

假设full=0.4，证据充分：

| 场景 | 遮挡系数O | 方向系数D | 新分数 |
| --- | ---: | ---: | ---: |
| 下方没有空扇区，1/2/5/6均纯净 | 1 | 1 | 0.400 |
| 下方至少一个空扇区，1/2/5/6均纯净 | 1.51 | 1 | 0.604 |
| 下方有空扇区，但任一方向扇区四方向等量 | 1.51 | exp(-0.6375)≈0.529 | 0.319 |
| 1/2/5/6任一扇区完全没有事件 | 1（G=0） | exp(-0.85)≈0.427 | 0.171 |

因此“缺失加分”和“混杂扣分”可能同时存在，最终净变化取决于两项乘积；空扇区不是无条件覆盖其他负面证据。

## 6. 调参入口与输入输出

参数全部集中于 `circle_detection/xiaoiron_confidence.py` 的 `XiaoironConfig`；Notebook参数cell有同样的显式配置。

| 参数 | 默认 | 调整含义 |
| --- | ---: | --- |
| occlusion_gain | 0.51 | 遮挡奖励强度；设0关闭奖励 |
| direction_penalty_lambda | 0.85 | 方向证据不足的惩罚强度；设0关闭扣分 |
| min_direction_events | 5 | 每个方向扇区足量的原始事件数 |
| min_direction_weight | 1.75 | 每个方向扇区足量的衰减权重 |
| min_other_sectors | 6 | 10/11/12/13以外的覆盖要求 |
| min_sector_weight | 1.0 | 有效覆盖扇区的最低权重 |
| occluded_sectors / direction_sectors | (10,11,12,13) / (1,2,5,6) | 参与计算的扇区组；新规则固定16等分 |

检测器调用方式不变：

```python
detector.push(event)                      # 每个事件入队
detection = detector.detect()             # 默认每6事件更新，其间返回上次结果
if detection is not None:
    old_score = detection.confidence      # 原完整圆几何分数
    new_score = detection.xiaoiron_confidence
    debug = detection.xiaoiron
    # debug.full_confidence / occlusion_factor / direction_factor
    # debug.evidence_strength / direction_evidence / direction_gap
    # debug.side_sufficiency / side_direction_evidence
    # debug.sector_counts / direction_weights
    # debug.sector_purity / sector_visible
```

纯评分函数 `calculate_xiaoiron_confidence` 不修改检测器状态，可以固定圆与事件窗口独立测试参数。
播放器缓存保留16×4直方图和原始计数，不只是最终标量；Notebook最后一个cell可按事件序号查看。
当前结构需要生成v6缓存。缓存生成后，播放器右侧调整公式参数不需要重跑几何检测。

## 7. 显示和验证

紫色曲线和圆默认使用新分数；Score按钮切回原评分，并保留full base作对照。
实时信息、各因子和16扇区方向柱形图置于事件图外。低于显示阈值时隐藏圆轮廓，但保留分数和扇区诊断；S可关闭候选辅助线。
柱形图对应评分事件窗口，事件图对应时间拖尾，二者可能包含不同事件。这不是漏播。

用当前默认参数完整处理155,584个真实事件后，有122,638个事件位置存在可用评分；其中11,974个位置
高于full、110,652个位置低于full。升降次数包含更新间隔内保持的结果，不是独立检测数，更不是准确率指标。
v5奖励样例为第126,703个事件（full 0.454，新分数0.615），最大扣分样例为第37,087个事件
（full 0.425，新分数0.182）。汇总记录在 `xiaoiron_validation.json`，对应PNG可直接查看。

下一步用人工标注的正常球、击打前后、遮挡、非球四类片段做对照；固定几何结果和阈值，比较旧分数与新分数的误报/漏报。
再分别将occlusion_gain和direction_penalty_lambda设0做消融。没有标注前不把默认值称作最优参数。

新评分不修改完整圆候选排序、接受门限及跟踪。原检测器完全拒绝的圆不会因为这项奖励自动复活。
新评分每次更新开销为O(W+16×4)，W为窗口事件数；默认每6事件一次。调试缓存另占O(N×16×4)存储，N为总事件数。

在本目录运行 `python -m unittest discover` 可执行单元测试；运行
`python scripts/verify_xiaoiron.py --project . --reuse-cache` 可复查 Notebook
计算与绘图 cell、缓存及两个真实数据样例。
去掉 `--reuse-cache` 会重新逐事件计算整个数据集。此验证脚本使用无界面的Agg后端，不弹出窗口。
