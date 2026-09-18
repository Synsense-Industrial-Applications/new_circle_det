# 离线工具

该 Python 包负责处理已经录制的 Layer-4 CSV：

- `event_stream_player.py`：只播放事件流，可分别查看 `0–7` 与 `8–15` 通道；
- `visualize_circle_detection.py`：预计算并播放圆检测结果；
- `four_region_flow.py`：把 `64×64×16` Layer-4 地址解码到 `128×128` 四方向事件；
- `runtime_performance_log.py`：记录离线播放器性能。

所有命令建议从项目根目录运行，具体示例见根目录 `README.md`。
