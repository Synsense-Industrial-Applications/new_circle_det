# Speck2f 台球事件流圆检测

本项目使用 Speck2f 上的 SNN 将 DVS 事件转换为 `64×64×16` 的 Layer-4
光流事件，再在 `128×128` 坐标系中进行圆心检测、置信度计算和击球回放。

## 核心入口

> **SNN 网络位于 [`algorithm_Demo/Demo_SNN.py`](algorithm_Demo/Demo_SNN.py)。**
>
> 该文件只包含当前实际运行的 Speck2f SNN 网络配置、各层连接、Layer-4
> `64×64×16` 输出和硬件可视化连接。圆检测算法及可执行主循环已独立到
> [`algorithm_Demo/Demo_algorithm.py`](algorithm_Demo/Demo_algorithm.py)。

其他常用入口：

| 文件 | 用途 |
|---|---|
| [`algorithm_Demo/Demo_SNN.py`](algorithm_Demo/Demo_SNN.py) | SNN 网络、层配置和 Layer-4 输出 |
| [`algorithm_Demo/Demo_algorithm.py`](algorithm_Demo/Demo_algorithm.py) | 实时圆检测、过滤、击球回放和主循环 |
| [`algorithm_Demo/Demo_record.py`](algorithm_Demo/Demo_record.py) | 录制 Layer-4，可选同步录制原始 DVS |
| [`algorithm_Demo/DS_Demo.py`](algorithm_Demo/DS_Demo.py) | 时间衰减 D/S 圆心检测实时 Demo |
| [`offline_tools/event_stream_player.py`](offline_tools/event_stream_player.py) | 轻量 Layer-4 事件播放器 |
| [`offline_tools/visualize_circle_detection.py`](offline_tools/visualize_circle_detection.py) | 离线圆检测、回放和参数调试 |
| [`notebooks/circle_detection_step_by_step.ipynb`](notebooks/circle_detection_step_by_step.ipynb) | 逐事件算法调试 Notebook |

## 目录结构

```text
new_circle_det/
├─ algorithm_Demo/             # SNN、Speck2f 实时程序和录制数据
│  ├─ Demo_SNN.py              # 当前 SNN 网络与硬件配置
│  ├─ Demo_algorithm.py        # 圆检测算法与实时运行入口
│  ├─ Demo_record.py           # Layer-4 / DVS 录制
│  ├─ DS_Demo.py               # D/S 圆心检测
│  ├─ assets/                  # 硬件 Demo 使用或参考的图片
│  ├─ legacy/                  # 不再由当前流程调用的旧实现
│  └─ recordings/              # 实机录制数据
├─ circle_detection/           # 圆候选、跟踪和置信度算法
├─ offline_tools/              # 离线解码、播放、检测和性能记录工具
├─ data/samples/               # 可直接运行的示例 Layer-4 数据
├─ notebooks/                  # Jupyter 调试流程
├─ tests/                      # offline 与 hardware 无硬件测试
├─ docs/                       # 公式说明和设计笔记
├─ scripts/                    # Windows 启动脚本与验证脚本
└─ artifacts/                  # 已生成的预览、视频和分析报告
```

根目录只保留项目说明、依赖文件和主要代码目录。生成结果不再与源码混放。
每个工程目录都包含自己的 `README.md`，进入任意目录即可查看其用途和维护说明。

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

仅使用轻量事件播放器时，可以只安装：

```bash
python -m pip install -r requirements_player.txt
```

Linux 若缺少 Tk 窗口支持，还需安装系统包 `python3-tk`。硬件程序另外需要
与 Speck2f 环境匹配的 `samna`、`samnagui` 和设备访问权限。

## 使用方法

所有命令均在项目根目录运行。

### 1. 运行 SNN 与实时圆检测

```bash
python algorithm_Demo/Demo_algorithm.py
```

该入口从 [`algorithm_Demo/Demo_SNN.py`](algorithm_Demo/Demo_SNN.py)
加载当前 SNN，再执行圆检测、过滤、日志和击球回放。网络和硬件参数说明见
[`algorithm_Demo/README.md`](algorithm_Demo/README.md)。

### 2. 录制 Layer-4 和 DVS

只保存 Layer-4：

```bash
python algorithm_Demo/Demo_record.py --no-record-dvs
```

同步保存 Layer-4 和原始 DVS：

```bash
python algorithm_Demo/Demo_record.py --record-dvs
```

按 `R` 开始或停止一段录制；再次按 `R` 会创建新文件；按 `Q` 退出。
录制结果保存在 `algorithm_Demo/recordings/`。

### 3. 播放事件数据

```bash
python offline_tools/event_stream_player.py data/samples/layer4_20260727_155031_part0001.csv --autoplay
```

播放器按照事件时间戳推进，支持暂停、调速、事件序号跳转、区间循环和时间渐隐。

### 4. 离线查看圆检测结果

```bash
python offline_tools/visualize_circle_detection.py
```

默认读取 `data/samples/` 中的示例数据。也可以指定其他文件：

```bash
python offline_tools/visualize_circle_detection.py --csv path/to/layer4.csv
```

Windows 也可以运行 [`scripts/run_circle_detection.bat`](scripts/run_circle_detection.bat)。

### 5. 运行 D/S 圆心检测

```bash
python algorithm_Demo/DS_Demo.py
```

例如调整检测迟滞与时间衰减参数：

```bash
python algorithm_Demo/DS_Demo.py --peak-threshold 0.14 --release-threshold 0.10 --decay-tau-ms 400
```

### 6. 使用 Notebook 调试

```bash
python -m pip install jupyter
jupyter notebook notebooks/circle_detection_step_by_step.ipynb
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

16 个 Layer-4 通道由两组互补空间卷积核组成，解码后合并为四个光流方向。
具体映射见 [`offline_tools/four_region_flow.py`](offline_tools/four_region_flow.py)。

## 测试

全部测试均不要求连接 Speck2f：

```bash
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

置信度公式和调参方法见
[`docs/XIAOIRON_CONFIDENCE.md`](docs/XIAOIRON_CONFIDENCE.md)。
