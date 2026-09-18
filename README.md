# Speck2f 台球事件流圆检测

本项目使用 Speck2f 上的 SNN 将 DVS 事件转换为 `64×64×16` 的 Layer-4
光流事件，并在 `128×128` 坐标系中进行台球圆心检测、置信度计算和击球回放。

项目同时支持：

- 连接 Speck2f 实时检测；
- 录制 Layer-4 与原始 DVS 事件；
- 离线播放、调参和验证录制数据；
- 使用时间衰减的 D/S 直方图实时估计圆心。

## 项目结构

```text
new_circle_det/
├─ algorithm_Demo/
│  ├─ Demo.py                 # Speck2f 实时圆检测与击球回放
│  ├─ Demo_record.py          # 录制 Layer-4，可选同步录制原始 DVS
│  ├─ DS_Demo.py              # 时间衰减 D/S 圆心检测 Demo
│  └─ README.md               # 硬件算法和参数说明
├─ circle_detection/          # 圆候选、跟踪和置信度算法
├─ event_stream_player.py     # 轻量 Layer-4 事件播放器
├─ visualize_circle_detection.py
│                              # 离线圆检测与参数调试播放器
├─ circle_detection_step_by_step.ipynb
│                              # 逐事件算法调试 Notebook
├─ four_region_flow.py        # 64×64×16 到 128×128 光流解码
└─ test_*.py                  # 离线单元测试
```

## 环境准备

推荐 Python 3.10 或更新版本。

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

仅使用轻量事件播放器时，可只安装最小依赖：

```bash
python -m pip install -r requirements_player.txt
```

Linux 若缺少 Tk 窗口支持，还需安装系统包 `python3-tk`。硬件 Demo 另外需要
与 Speck2f 环境匹配的 `samna`、`samnagui` 和设备访问权限；这些硬件依赖没有
写入通用 `requirements.txt`。

## 快速使用

### 1. 播放已录制的 Layer-4 数据

```bash
python event_stream_player.py path/to/layer4.csv --autoplay
```

也可以不传文件路径，启动后点击“打开 CSV”。播放器按照事件时间戳推进，支持
暂停、调速、事件序号跳转、区间循环和时间渐隐。

### 2. 离线查看圆检测结果

```bash
python visualize_circle_detection.py --csv path/to/layer4.csv
```

界面可实时调整显示阈值和 `xiaoiron_confidence` 参数。要逐事件修改算法，请先安装
Jupyter，再打开：

```bash
python -m pip install jupyter
jupyter notebook circle_detection_step_by_step.ipynb
```

### 3. 运行 Speck2f 实时检测

连接设备并启动 `samnagui` 环境后，在项目根目录运行：

```bash
python algorithm_Demo/Demo.py
```

详细的数据流、过滤条件和击球回放参数见
[algorithm_Demo/README.md](algorithm_Demo/README.md)。

### 4. 录制数据

只录制 Layer-4（默认不保存原始 DVS）：

```bash
python algorithm_Demo/Demo_record.py --no-record-dvs
```

同步保存 Layer-4 和原始 DVS：

```bash
python algorithm_Demo/Demo_record.py --record-dvs
```

运行后按 `R` 开始或停止一段录制，再次按 `R` 会创建新文件；按 `Q` 退出。
文件保存在 `algorithm_Demo/recordings/` 下。

### 5. 运行 D/S 圆心检测 Demo

```bash
python algorithm_Demo/DS_Demo.py
```

常用参数可直接从命令行调整，例如：

```bash
python algorithm_Demo/DS_Demo.py --peak-threshold 0.14 --release-threshold 0.10 --decay-tau-ms 400
```

## 数据格式

Layer-4 CSV：

```text
x,y,feature,timestamp
```

- `x, y`：`64×64` Layer-4 地址；
- `feature`：`0..15` 输出通道；
- `timestamp`：事件时间戳。

原始 DVS CSV：

```text
x,y,polarity,timestamp
```

Layer-4 的 16 个通道由两组互补空间卷积核组成，解码后合并为四个光流方向。
具体映射位于 [four_region_flow.py](four_region_flow.py)。

## 测试

运行根目录离线测试：

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

运行硬件 Demo 的无硬件单元测试：

```bash
python -m unittest discover -s algorithm_Demo -p "test_*.py" -v
```

置信度公式和调参说明见
[XIAOIRON_CONFIDENCE.md](XIAOIRON_CONFIDENCE.md)。
