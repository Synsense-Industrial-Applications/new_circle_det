import csv
import os
import threading
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty, Full, Queue as ThreadQueue

import numpy as np


class point_withchannel:
    def __init__(self, c, x, y):
        self.c = c
        self.x = x
        self.y = y


class ChannelHelper:
    """通用 channel/坐标映射工具，和具体 Speck 网络配置解耦。"""

    def __init__(self, input_channel, input_size, output_size):
        self.input_channel = input_channel
        self.input_size = input_size
        self.output_size = output_size

    def get_tuple(self, input_tuple, mode_flage=0):
        c, x, y = input_tuple
        if mode_flage == 0:
            if self.input_size == self.output_size:
                return (c, x, y)
            if self.input_size > self.output_size:
                ratio = self.input_size // self.output_size
                local_pos = (x % ratio) * ratio + (y % ratio)
                out_c = c * (ratio**2) + local_pos
                out_x = x // ratio
                out_y = y // ratio
                return (out_c, out_x, out_y)

            ratio = self.output_size // self.input_size
            out_c = c // (ratio**2)
            local_pos = c % (ratio**2)
            j = local_pos // ratio
            k = local_pos % ratio
            out_x = x * ratio + j
            out_y = y * ratio + k
            return (out_c, out_x, out_y)

        if mode_flage == 1:
            if self.input_size == self.output_size:
                return (c, x, y)
            if self.input_size > self.output_size:
                ratio = self.input_size // self.output_size
                in_c = c // (ratio**2)
                local_pos = c % (ratio**2)
                j = local_pos // ratio
                k = local_pos % ratio
                in_x = x * ratio + j
                in_y = y * ratio + k
                return (in_c, in_x, in_y)

            ratio = self.output_size // self.input_size
            local_pos = (x % ratio) * ratio + (y % ratio)
            in_c = c * (ratio**2) + local_pos
            in_x = x // ratio
            in_y = y // ratio
            return (in_c, in_x, in_y)

        raise ValueError(f"unknown mode_flage: {mode_flage}")

    def generate_conv_mapping(self):
        if self.input_size == self.output_size:
            raise ValueError("Input and output sizes are equal; no mapping needed.")

        if self.input_size > self.output_size:
            ratio = self.input_size // self.output_size
            output_channel = self.input_channel * (ratio**2)
            padding = 0
            stride = ratio
            kernel_size = ratio
            weights = np.zeros((output_channel, self.input_channel, kernel_size, kernel_size), dtype=np.int8)

            for in_c in range(self.input_channel):
                for j in range(ratio):
                    for k in range(ratio):
                        out_c = in_c * (ratio**2) + j * ratio + k
                        weights[out_c, in_c, j, k] = 1
            return padding, stride, kernel_size, weights

        ratio = self.output_size // self.input_size
        output_channel = self.input_channel // (ratio**2)
        padding = 0
        stride = 1
        kernel_size = ratio
        weights = np.zeros((output_channel, self.input_channel, kernel_size, kernel_size), dtype=np.int8)

        for out_c in range(output_channel):
            for j in range(ratio):
                for k in range(ratio):
                    in_c = out_c * (ratio**2) + j * ratio + k
                    weights[out_c, in_c, j, k] = 1
        return padding, stride, kernel_size, weights

    def caculate_cxy(self, x, y, radius):
        input_size = self.input_size
        output_size = self.output_size
        out_channel = input_size // output_size
        ret_x = (x - radius) // out_channel
        ret_y = (y - radius) // out_channel
        c = 0

        if x - radius < 0:
            remainder_x = np.abs(x - radius) % out_channel
            c += 0 if remainder_x == 0 else (out_channel - remainder_x) * out_channel
        else:
            c += ((x - radius) % out_channel) * out_channel

        if y - radius < 0:
            remainder_y = np.abs(y - radius) % out_channel
            c += 0 if remainder_y == 0 else (out_channel - remainder_y)
        else:
            c += (y - radius) % out_channel

        return c, ret_x, ret_y

    def get_kernel(self, kernel_matrix, size=None, drop=False):
        """把 input_size 空间的内核映射到 output_size 空间。

        返回 (out_channel**2, out_channel**2, shift_kernel_size, shift_kernel_size)。
        索引 = center + 偏移，center = (shift_kernel_size - 1) // 2，
        卷积层 padding 取同样的 center（脚本里就是 (shape[2] - 1) // 2）。

        size=None（默认）：尺寸 = 2 * ceil(radius / out_channel) + 1，只取决于
        kernel_size，所以同一层里各个子内核形状一致（脚本里 5x5 -> 3x3）。
        size=S：强制用 S，要求所有非零 tap 都放得下（放不下直接报错，不会静默丢权重）。
        drop=True 时改为“放不下就丢掉该 tap”，并打印丢掉的个数（用于最小尺寸的近似版本）。
        128 空间 tap 在 dx = -1 时，偶目标落到 coarse -1、奇目标落到 0；dx = +1 时
        分别是 0 和 +1。所以图案在 ±1 两侧都有 tap 时必须用 3 格；把图案整体平移到
        dx >= 0 的单侧后只需要 {0,+1}，可以显式传 size=2 得到 4x4x2x2 的最小内核。
        尺寸为偶数时 conv 输出比输入少 1，硬件要求
        out = floor((in - kernel + 2 * padding) / stride) + 1 且 out <= 64。
        """
        input_size = self.input_size
        output_size = self.output_size
        out_channel = input_size // output_size
        kernel_size = len(kernel_matrix)
        radius = kernel_size // 2
        caculate_size = out_channel + radius * 2

        adjacency_matrix = [
            [point_withchannel(*self.caculate_cxy(x, y, radius)) for y in range(caculate_size)]
            for x in range(caculate_size)
        ]

        if size is None:
            center = -(-radius // out_channel)             # ceil(radius / out_channel)
            shift_kernel_size = center * 2 + 1
        else:
            shift_kernel_size = int(size)
            center = (shift_kernel_size - 1) // 2
        kernel_weight = np.zeros((out_channel**2, out_channel**2, shift_kernel_size, shift_kernel_size))
        dropped = 0
        for i in range(out_channel**2):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    if kernel_matrix[j][k] == 0:
                        continue
                    temp_x = radius + i // out_channel
                    temp_y = radius + i % out_channel
                    source = adjacency_matrix[temp_x - (radius - j)][temp_y - (radius - k)]
                    target = adjacency_matrix[temp_x][temp_y]
                    weight_x = center + source.x - target.x
                    weight_y = center + source.y - target.y
                    if not (0 <= weight_x < shift_kernel_size and 0 <= weight_y < shift_kernel_size):
                        if drop:
                            dropped += 1
                            continue
                        raise AssertionError(
                            f"kernel_size {kernel_size} 的 tap ({j},{k}) 需要 coarse 偏移 "
                            f"({weight_x - center},{weight_y - center})，放不进 {shift_kernel_size}x"
                            f"{shift_kernel_size}（size={size}）")
                    kernel_weight[i, source.c, weight_x, weight_y] = kernel_matrix[j][k]
        print(f"kernel_size: {kernel_size} -> {shift_kernel_size}, center {center}"
              + (f", dropped {dropped} taps" if drop else ""))
        return kernel_weight

    def convert_matrix_to_128(self, matrix_64, out_channel):
        return matrix_64[out_channel, :, :, :]

    def get_conv_shape(self):
        return self.input_channel, self.output_size, self.output_size


def write_events_to_csv(events, folder_path):
    """将 samna 事件按 physical layer 追加写入 layer_*.csv。"""
    os.makedirs(folder_path, exist_ok=True)

    layer_events = {}
    for event in events:
        layer_events.setdefault(event.layer, []).append(event)

    for layer, events_in_layer in layer_events.items():
        filepath = os.path.join(folder_path, f"layer_{layer}.csv")
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["feature", "y", "x", "timestamp"])
            for event in events_in_layer:
                writer.writerow([event.feature, event.y, event.x, event.timestamp])


def background_writer(queue, folder_path):
    """后台 CSV 写入进程入口。当前主脚本默认不用后台进程。"""
    while True:
        events = queue.get()
        if events is None:
            break
        write_events_to_csv(events, folder_path)


def start_background_writer(folder_path):
    queue = Queue()
    process = Process(target=background_writer, args=(queue, folder_path))
    process.daemon = True
    process.start()
    return queue, process


def print_layer_feature_counts(events, watched_layers=None):
    """按 layer/feature 打印一批事件的数量。"""
    watched_layers = set(watched_layers) if watched_layers is not None else None
    layer_feature_counts = defaultdict(lambda: defaultdict(int))
    layer_counts = defaultdict(int)

    for event in events:
        if watched_layers is not None and event.layer not in watched_layers:
            continue
        layer_feature_counts[event.layer][event.feature] += 1
        layer_counts[event.layer] += 1

    for layer in sorted(layer_feature_counts):
        print(f"  layer {layer}: {layer_counts[layer]} events")
        for feature in sorted(layer_feature_counts[layer]):
            print(f"    feature {feature}: {layer_feature_counts[layer][feature]}")
    print(f"  total: {sum(layer_counts.values())} events\n")


def store_events_to_csv(events, folder_path):
    """保存传入的事件。注意：不要在保存前重复调用 buf.get_events()。"""
    if events:
        write_events_to_csv(events, folder_path)
        print(f"  [store] {len(events)} events saved to {folder_path}")


def reset_data_folder(folder_path, base_dir=None):
    """删除数据目录中的旧 layer_*.csv，并限制清理范围在当前工作区内。"""
    abs_folder = os.path.abspath(folder_path)
    abs_base = os.path.abspath(base_dir or os.getcwd())
    if abs_folder != abs_base and not abs_folder.startswith(abs_base + os.sep):
        raise ValueError(f"拒绝清理工作区外的数据目录: {abs_folder}")

    os.makedirs(abs_folder, exist_ok=True)
    removed = 0
    for name in os.listdir(abs_folder):
        if name.startswith("layer_") and name.endswith(".csv"):
            os.remove(os.path.join(abs_folder, name))
            removed += 1
    print(f"[reset] 已删除 {removed} 个旧 layer CSV 文件: {abs_folder}")


class RealtimeEventVisualizer:
    """用后台线程播放最新一批事件。

    主线程只负责提交事件；播放线程负责显示。队列容量固定为 1，
    新数据进入时会丢弃还没开始播放的旧数据，正在播放的旧数据也会在下一帧被打断。
    """

    def __init__(
        self,
        layers=None,
        title="Realtime events",
        size=(64, 64),
        hold_ms=80.0,
        fps=30,
        point_size=8.0,
        channel_names=None,
        channel_colors=None,
        drop_old=True,
        enabled=True,
    ):
        self.layers = set(layers) if layers is not None else None
        self.title = title
        self.width, self.height = size
        self.hold_us = int(round(hold_ms * 1000.0))
        self.fps = max(int(fps), 1)
        self.point_size = point_size
        self.channel_names = channel_names or {}
        self.channel_colors = channel_colors or {
            0: "#d62728",
            1: "#1f77b4",
            2: "#2ca02c",
            3: "#ff7f0e",
            4: "#9467bd",
            5: "#17becf",
            6: "#8c564b",
            7: "#e377c2",
        }
        self.drop_old = drop_old
        self.enabled = enabled
        self._plt = None
        self._fig = None
        self._ax = None
        self._image_artist = None
        self._blank_image = None
        self._frame_image = None
        self._color_table = None
        self._to_rgb = None
        self._last_submit_time = None
        self._queue = ThreadQueue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None
        self._worker_lock = threading.Lock()

    def _ensure_window(self):
        if not self.enabled or self._fig is not None:
            return
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from matplotlib.lines import Line2D

        self._plt = plt
        self._to_rgb = to_rgb
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(5.5, 5.5))
        self._fig.canvas.manager.set_window_title(self.title)
        self._ax.set_title(self.title)
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

        self._blank_image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self._frame_image = np.zeros_like(self._blank_image)
        self._image_artist = self._ax.imshow(
            self._blank_image,
            origin="upper",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        self._ax.set_xlim(-0.5, self.width - 0.5)
        self._ax.set_ylim(self.height - 0.5, -0.5)
        self._ax.set_aspect("equal", adjustable="box")

        self._color_table = {}
        for feature, color in self.channel_colors.items():
            self._color_table[int(feature)] = np.array(to_rgb(color), dtype=np.float32)

        handles = []
        for feature in sorted(self.channel_names):
            color = self.channel_colors.get(feature, "#333333")
            label = self.channel_names.get(feature, f"ch {feature}")
            handles.append(Line2D([0], [0], marker="s", color=color, linestyle="", label=label, markersize=6))
        if handles:
            self._ax.legend(handles=handles, loc="upper right", fontsize=8)

        self._fig.tight_layout()
        plt.show(block=False)

    def start(self):
        """启动后台播放线程。"""
        if not self.enabled:
            return
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._worker_loop, name=f"{self.title} player", daemon=True)
            self._worker.start()

    def stop(self):
        """停止后台播放线程。"""
        self._stop_event.set()
        self._drop_queued_items()

    def _drop_queued_items(self):
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _put_latest(self, item):
        if self.drop_old:
            self._drop_queued_items()
        try:
            self._queue.put_nowait(item)
            return
        except Full:
            pass

        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except Full:
            pass

    def _extract_events(self, events):
        rows = []
        for event in events:
            if self.layers is not None and event.layer not in self.layers:
                continue
            rows.append((int(event.timestamp), int(event.feature), int(event.x), int(event.y), int(event.layer)))
        if not rows:
            return None
        return np.array(rows, dtype=np.int64)

    def _draw_frame(self, frame_events, batch_index, batch_count, duration_s):
        image = self._frame_image
        image.fill(0.0)
        if len(frame_events):
            features = sorted(np.unique(frame_events[:, 1]).tolist())
            for feature in features:
                sub = frame_events[frame_events[:, 1] == feature]
                color = self._color_table.get(feature)
                if color is None:
                    color = np.array(self._to_rgb("#333333"), dtype=np.float32)
                    self._color_table[feature] = color

                xs = sub[:, 2]
                ys = sub[:, 3]
                valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
                image[ys[valid], xs[valid]] = color

        self._image_artist.set_data(image)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def play_events(self, events, playback_duration_s=None):
        """提交一批新事件给后台线程，旧数据会被新数据抢占。"""
        if not self.enabled:
            return
        now = time.perf_counter()
        if playback_duration_s is None:
            if self._last_submit_time is None:
                playback_duration_s = 1.0
            else:
                playback_duration_s = now - self._last_submit_time
        self._last_submit_time = now

        self.start()
        self._put_latest((events, max(float(playback_duration_s), 0.001)))

    def _prepare_playback(self, events, playback_duration_s):
        arr = self._extract_events(events)
        frame_count = max(1, int(np.ceil(playback_duration_s * self.fps)))
        if arr is None:
            arr = np.empty((0, 5), dtype=np.int64)

        # 按事件数量均匀摊到本轮采集间隔中，不使用事件自身 timestamp 展开。
        frame_edges = np.linspace(0, len(arr), frame_count + 1, dtype=np.int64)
        hold_frames = max(1, int(np.ceil((self.hold_us / 1_000_000.0) * self.fps)))

        return {
            "events": arr,
            "frame_edges": frame_edges,
            "duration_s": playback_duration_s,
            "frame_count": frame_count,
            "hold_frames": hold_frames,
        }

    def update(self):
        """兼容旧调用方式；现在刷新由后台线程负责。"""
        self.start()

    def _worker_loop(self):
        self._ensure_window()
        pending_item = None
        while not self._stop_event.is_set():
            if pending_item is None:
                try:
                    pending_item = self._queue.get(timeout=0.02)
                except Empty:
                    self._plt.pause(0.001)
                    continue

            events, playback_duration_s = pending_item
            pending_item = None
            playback = self._prepare_playback(events, playback_duration_s)
            start_wall = time.perf_counter()
            last_drawn_frame = None

            while not self._stop_event.is_set():
                # 有新数据时立刻中断当前批次，下一轮直接播放最新数据。
                try:
                    pending_item = self._queue.get_nowait()
                    break
                except Empty:
                    pass

                duration_s = playback["duration_s"]
                frame_count = playback["frame_count"]
                elapsed_s = time.perf_counter() - start_wall
                frame_index = int(elapsed_s / duration_s * frame_count)
                frame_index = min(max(frame_index, 0), frame_count - 1)

                if frame_index != last_drawn_frame:
                    arr = playback["events"]
                    frame_edges = playback["frame_edges"]
                    hold_frames = playback["hold_frames"]
                    left_frame = max(0, frame_index - hold_frames + 1)
                    left_idx = frame_edges[left_frame]
                    right_idx = frame_edges[frame_index + 1]
                    frame_events = arr[left_idx:right_idx]
                    self._draw_frame(frame_events, frame_index + 1, frame_count, duration_s)
                    last_drawn_frame = frame_index

                if elapsed_s >= duration_s:
                    break

                next_frame_wall = start_wall + (frame_index + 1) * duration_s / frame_count
                pause_s = min(max(next_frame_wall - time.perf_counter(), 0.001), 0.02)
                self._plt.pause(pause_s)
