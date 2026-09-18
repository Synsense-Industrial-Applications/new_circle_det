# 圆检测核心算法

该 Python 包包含与硬件接口无关的圆检测逻辑：

- `detector.py`：方向约束的基础检测器；
- `adaptive_detector.py`：圆候选生成、细化、跟踪和事件窗口管理；
- `xiaoiron_confidence.py`：遮挡和方向一致性置信度。

实时 Demo 和离线播放器共用这里的算法。修改后应运行 `tests/offline/` 和
`tests/hardware/` 中的全部测试。
