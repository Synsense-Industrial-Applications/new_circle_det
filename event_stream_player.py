"""Lightweight Layer-4 event-stream player.

This file intentionally contains no circle detector, confidence formula,
online scorer, Pillow, or Matplotlib dependency.  It only needs NumPy and the
Python standard-library Tk interface.

Features:
* load raw ``x,y,feature,timestamp`` Layer-4 CSV files;
* decode 64x64x8 addresses to the 128x128 four-direction view;
* timestamp-accurate playback from 0.01x to 20x;
* play/pause, restart, previous/next event and exact event-number jump;
* draggable progress, loop range, event-density timeline;
* configurable trail, exponential time fading and glow intensity.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np


IMAGE_SIZE = 128
PLAYBACK_TICK_MS = 16
SPEED_OPTIONS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
)
FLOW_DIRECTION_NAMES = ("↘", "↙", "↖", "↗")
FLOW_DIRECTION_COLORS = ("#ff453a", "#ffd60a", "#0a84ff", "#30d158")
FLOW_DIRECTION_RGB = np.asarray(
    (
        (255.0, 69.0, 58.0),
        (255.0, 214.0, 10.0),
        (10.0, 132.0, 255.0),
        (48.0, 209.0, 88.0),
    ),
    dtype=np.float32,
)
FEATURE_TO_DIRECTION = np.asarray(
    (0, 1, 1, 0, 2, 3, 3, 2),
    dtype=np.int8,
)
ARROW_VECTORS = (
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
GLOW_KERNEL = (
    (0, 0, 1.00),
    (-1, 0, 0.34),
    (1, 0, 0.34),
    (0, -1, 0.34),
    (0, 1, 0.34),
    (-1, -1, 0.12),
    (1, -1, 0.12),
    (-1, 1, 0.12),
    (1, 1, 0.12),
)


def set_windows_timer_resolution(enable: bool) -> bool:
    """Request a 1 ms Windows timer; harmlessly does nothing elsewhere."""

    if sys.platform != "win32":
        return False
    try:
        winmm = ctypes.windll.winmm
        function = winmm.timeBeginPeriod if enable else winmm.timeEndPeriod
        return function(1) == 0
    except (AttributeError, OSError):
        return False


def _field(fields, *names):
    for name in names:
        if name in fields:
            return fields[name]
    return None


def load_event_csv(path):
    """Load and stably time-sort raw or already decoded event CSV data."""

    path = Path(path)
    timestamps = []
    xs = []
    ys = []
    features = []

    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = {
            str(name).strip().lower(): name
            for name in (reader.fieldnames or ())
        }
        timestamp_field = _field(
            fields,
            "event_timestamp",
            "event_timestamp_us",
            "timestamp",
            "timestamps",
            "t",
        )
        x_field = _field(fields, "x", "event_x")
        y_field = _field(fields, "y", "event_y")
        feature_field = _field(fields, "feature", "event_feature")
        if timestamp_field is None or x_field is None or y_field is None:
            raise ValueError("CSV must contain timestamp, x and y columns")

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamps.append(int(float(row[timestamp_field])))
                xs.append(int(float(row[x_field])))
                ys.append(int(float(row[y_field])))
                if feature_field is not None:
                    features.append(int(float(row[feature_field])))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid CSV value at row {row_number}: {error}"
                ) from error

    if not timestamps:
        raise ValueError("Event CSV is empty")

    timestamps = np.asarray(timestamps, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int16)
    ys = np.asarray(ys, dtype=np.int16)
    directions = np.zeros(len(timestamps), dtype=np.int8)
    source_shape = "128x128"

    if features:
        features = np.asarray(features, dtype=np.int16)
        if len(features) != len(timestamps):
            raise ValueError("feature must be present on every CSV row")
        if np.any((features < 0) | (features > 7)):
            raise ValueError("Layer-4 feature must be in 0..7")
        directions = FEATURE_TO_DIRECTION[features]

        if (
            np.all((xs >= 0) & (xs < 64))
            and np.all((ys >= 0) & (ys < 64))
        ):
            raw_x = xs.astype(np.int32)
            raw_y = ys.astype(np.int32)
            feature_mod4 = features.astype(np.int32) % 4
            xs = (raw_x * 2 + (feature_mod4 // 2) % 2).astype(np.int16)
            ys = (raw_y * 2 + feature_mod4 % 2).astype(np.int16)
            source_shape = "64x64x8 -> 128x128 four-direction flow"

    invalid = (xs < 0) | (xs >= IMAGE_SIZE) | (ys < 0) | (ys >= IMAGE_SIZE)
    if np.any(invalid):
        index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"Event {index + 1} is outside 0..127: "
            f"({int(xs[index])}, {int(ys[index])})"
        )

    order = np.argsort(timestamps, kind="stable")
    return (
        timestamps[order],
        xs[order],
        ys[order],
        directions[order],
        source_shape,
    )


def build_event_rgb(
    timestamps,
    xs,
    ys,
    directions,
    index,
    current_timestamp,
    trail_us,
    fade_tau_us,
    gain,
):
    """Render one 128x128 RGB frame and return it with the visible count."""

    current_timestamp = float(current_timestamp)
    first = int(
        np.searchsorted(timestamps, current_timestamp - trail_us, side="left")
    )
    recent = slice(first, index + 1)
    age_us = current_timestamp - timestamps[recent]
    weights = np.exp(-age_us / max(float(fade_tau_us), 1.0)).astype(
        np.float32
    )
    recent_x = xs[recent].astype(np.int32)
    recent_y = ys[recent].astype(np.int32)
    recent_directions = directions[recent].astype(np.int32)

    heat = np.zeros((4, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for dx, dy, kernel_weight in GLOW_KERNEL:
        target_x = recent_x + dx
        target_y = recent_y + dy
        valid = (
            (target_x >= 0)
            & (target_x < IMAGE_SIZE)
            & (target_y >= 0)
            & (target_y < IMAGE_SIZE)
        )
        if np.any(valid):
            np.add.at(
                heat,
                (
                    recent_directions[valid],
                    target_y[valid],
                    target_x[valid],
                ),
                weights[valid] * kernel_weight,
            )

    intensity = 1.0 - np.exp(-max(float(gain), 0.01) * heat)
    rgb = np.einsum(
        "dyx,dc->yxc",
        intensity,
        FLOW_DIRECTION_RGB,
        optimize=True,
    )
    rgb += np.asarray((3.0, 6.0, 10.0), dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8), index - first + 1


def format_duration(value_us):
    seconds = max(0.0, float(value_us) * 1e-6)
    minutes, seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes:02d}:{seconds:06.3f}"


class EventStreamPlayer:
    def __init__(
        self,
        root,
        event_path=None,
        speed=1.0,
        persistence_ms=160.0,
        fade_tau_ms=50.0,
        gain=1.0,
        autoplay=False,
    ):
        self.root = root
        self.event_path = None
        self.timestamps = np.asarray([], dtype=np.int64)
        self.xs = np.asarray([], dtype=np.int16)
        self.ys = np.asarray([], dtype=np.int16)
        self.directions = np.asarray([], dtype=np.int8)
        self.source_shape = ""
        self.index = 0
        self.playback_timestamp = 0.0
        self.playing = False
        self.autoplay = bool(autoplay)
        self.speed = min(SPEED_OPTIONS, key=lambda value: abs(value - speed))
        self.anchor_wall_time = time.perf_counter()
        self.anchor_timestamp = 0.0
        self.updating_progress = False
        self.visible_count = 0
        self.high_resolution_timer = set_windows_timer_resolution(True)

        self.base_photo = None
        self.scaled_photo = None
        self.photo_scale = None
        self.image_photo = None
        self.resize_job = None
        self.timeline_resize_job = None
        self.density_counts = np.asarray([], dtype=np.float64)
        self.density_edges_us = np.asarray([], dtype=np.float64)
        self.timeline_cursor = None

        self.speed_var = tk.StringVar(value=self._speed_text(self.speed))
        self.trail_var = tk.DoubleVar(value=max(5.0, float(persistence_ms)))
        self.fade_var = tk.DoubleVar(value=max(1.0, float(fade_tau_ms)))
        self.gain_var = tk.DoubleVar(value=max(0.05, float(gain)))
        self.trail_text = tk.StringVar()
        self.fade_text = tk.StringVar()
        self.gain_text = tk.StringVar()
        self.info_var = tk.StringVar(value="请打开一份 Layer-4 CSV。")
        self.status_var = tk.StringVar(value="尚未加载文件")
        self.event_number_var = tk.StringVar(value="1")
        self.progress_text_var = tk.StringVar(value="event 0 / 0")
        self.loop_enabled_var = tk.BooleanVar(value=False)
        self.loop_start_var = tk.StringVar(value="1")
        self.loop_end_var = tk.StringVar(value="1")

        self.root.title("Layer-4 事件流播放器")
        self.root.geometry("1180x820")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._bind_keys()
        self._update_display_labels()
        self._set_loaded_state(False)
        self.root.after(PLAYBACK_TICK_MS, self.tick)

        if event_path is not None:
            self.root.after(50, lambda: self.load_file(Path(event_path)))
        else:
            self.root.after(100, self.choose_file)

    @staticmethod
    def _speed_text(speed):
        return f"{speed:g}x"

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        self.open_button = ttk.Button(
            toolbar, text="打开 CSV", command=self.choose_file
        )
        self.open_button.pack(side=tk.LEFT)
        self.restart_button = ttk.Button(
            toolbar, text="重新开始", command=self.restart
        )
        self.restart_button.pack(side=tk.LEFT, padx=(12, 4))
        self.previous_button = ttk.Button(
            toolbar, text="上一事件", command=lambda: self.step_event(-1)
        )
        self.previous_button.pack(side=tk.LEFT, padx=4)
        self.play_button = ttk.Button(
            toolbar, text="播放", command=self.toggle_play
        )
        self.play_button.pack(side=tk.LEFT, padx=4)
        self.next_button = ttk.Button(
            toolbar, text="下一事件", command=lambda: self.step_event(1)
        )
        self.next_button.pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="速度").pack(side=tk.LEFT, padx=(22, 6))
        self.speed_combo = ttk.Combobox(
            toolbar,
            width=8,
            state="readonly",
            textvariable=self.speed_var,
            values=[self._speed_text(value) for value in SPEED_OPTIONS],
        )
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)
        ttk.Label(
            toolbar,
            text="空格：播放/暂停   ←/→：单事件   +/-：调速",
        ).pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            body,
            background="#03060a",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        sidebar = ttk.Frame(body, padding=(14, 0, 0, 0), width=270)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)

        info_frame = ttk.LabelFrame(sidebar, text="事件信息", padding=10)
        info_frame.pack(fill=tk.X)
        ttk.Label(
            info_frame,
            textvariable=self.info_var,
            justify=tk.LEFT,
            anchor="nw",
        ).pack(fill=tk.X)

        legend_frame = ttk.LabelFrame(sidebar, text="光流方向", padding=10)
        legend_frame.pack(fill=tk.X, pady=(10, 0))
        for row, (name, color) in enumerate(
            zip(FLOW_DIRECTION_NAMES, FLOW_DIRECTION_COLORS)
        ):
            swatch = tk.Label(legend_frame, background=color, width=3)
            swatch.grid(row=row, column=0, padx=(0, 8), pady=3)
            ttk.Label(legend_frame, text=f"c={row}   {name}").grid(
                row=row, column=1, sticky="w"
            )

        display_frame = ttk.LabelFrame(sidebar, text="显示效果", padding=10)
        display_frame.pack(fill=tk.X, pady=(10, 0))
        self._add_scale(
            display_frame,
            0,
            "拖尾范围",
            self.trail_var,
            self.trail_text,
            5.0,
            1000.0,
        )
        self._add_scale(
            display_frame,
            1,
            "渐隐时间常数",
            self.fade_var,
            self.fade_text,
            1.0,
            500.0,
        )
        self._add_scale(
            display_frame,
            2,
            "亮度增益",
            self.gain_var,
            self.gain_text,
            0.05,
            8.0,
        )

        loop_frame = ttk.LabelFrame(sidebar, text="区间循环", padding=10)
        loop_frame.pack(fill=tk.X, pady=(10, 0))
        self.loop_check = ttk.Checkbutton(
            loop_frame,
            text="循环播放选定区间",
            variable=self.loop_enabled_var,
            command=self.on_loop_toggle,
        )
        self.loop_check.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(loop_frame, text="起点").grid(row=1, column=0, pady=(8, 0))
        self.loop_start_entry = ttk.Entry(
            loop_frame, width=9, textvariable=self.loop_start_var
        )
        self.loop_start_entry.grid(row=1, column=1, pady=(8, 0), padx=5)
        ttk.Label(loop_frame, text="终点").grid(row=2, column=0, pady=(5, 0))
        self.loop_end_entry = ttk.Entry(
            loop_frame, width=9, textvariable=self.loop_end_var
        )
        self.loop_end_entry.grid(row=2, column=1, pady=(5, 0), padx=5)
        self.loop_apply_button = ttk.Button(
            loop_frame, text="应用", command=self.apply_loop_range
        )
        self.loop_apply_button.grid(row=1, column=2, rowspan=2, padx=(5, 0))

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.timeline = tk.Canvas(
            bottom,
            height=86,
            background="#09101a",
            highlightthickness=0,
        )
        self.timeline.grid(row=0, column=0, columnspan=8, sticky="ew")
        self.timeline.bind("<Configure>", self.on_timeline_resize)
        self.timeline.bind("<Button-1>", self.on_timeline_click)

        self.progress_scale = ttk.Scale(
            bottom,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.on_progress,
        )
        self.progress_scale.grid(row=1, column=0, columnspan=4, sticky="ew")
        ttk.Label(bottom, textvariable=self.progress_text_var).grid(
            row=1, column=4, padx=(12, 16)
        )
        ttk.Label(bottom, text="事件序号").grid(row=1, column=5)
        self.event_entry = ttk.Entry(
            bottom, width=10, textvariable=self.event_number_var
        )
        self.event_entry.grid(row=1, column=6, padx=(5, 4))
        self.event_entry.bind("<Return>", self.jump_to_event)
        self.jump_button = ttk.Button(
            bottom, text="跳转", command=self.jump_to_event
        )
        self.jump_button.grid(row=1, column=7)

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=(10, 2, 10, 8),
        ).grid(row=3, column=0, sticky="ew")

    def _add_scale(self, parent, row, title, variable, text_variable, start, end):
        ttk.Label(parent, text=title).grid(row=row * 2, column=0, sticky="w")
        ttk.Label(parent, textvariable=text_variable).grid(
            row=row * 2, column=1, sticky="e"
        )
        scale = ttk.Scale(
            parent,
            from_=start,
            to=end,
            variable=variable,
            command=self.on_display_change,
        )
        scale.grid(row=row * 2 + 1, column=0, columnspan=2, sticky="ew")
        parent.columnconfigure(0, weight=1)

    def _bind_keys(self):
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.step_event(-1))
        self.root.bind("<Right>", lambda _event: self.step_event(1))
        self.root.bind("<Home>", lambda _event: self.restart())
        self.root.bind("<Control-o>", lambda _event: self.choose_file())
        self.root.bind("<plus>", lambda _event: self.change_speed(1))
        self.root.bind("<equal>", lambda _event: self.change_speed(1))
        self.root.bind("<minus>", lambda _event: self.change_speed(-1))

    def _set_loaded_state(self, loaded):
        state = "normal" if loaded else "disabled"
        for widget in (
            self.restart_button,
            self.previous_button,
            self.play_button,
            self.next_button,
            self.progress_scale,
            self.event_entry,
            self.jump_button,
            self.loop_check,
            self.loop_start_entry,
            self.loop_end_entry,
            self.loop_apply_button,
        ):
            widget.configure(state=state)

    def choose_file(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="打开 Layer-4 事件 CSV",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path):
        try:
            (
                timestamps,
                xs,
                ys,
                directions,
                source_shape,
            ) = load_event_csv(path)
        except Exception as error:
            messagebox.showerror("无法打开 CSV", str(error), parent=self.root)
            return

        self.event_path = Path(path)
        self.timestamps = timestamps
        self.xs = xs
        self.ys = ys
        self.directions = directions
        self.source_shape = source_shape
        self.index = 0
        self.playback_timestamp = float(self.timestamps[0])
        self.playing = False
        self.loop_enabled_var.set(False)
        self.loop_start_var.set("1")
        self.loop_end_var.set(str(len(self.timestamps)))
        duration_us = max(1.0, float(self.timestamps[-1] - self.timestamps[0]))
        self.progress_scale.configure(from_=0.0, to=duration_us)
        self._build_density_profile()
        self._set_loaded_state(True)
        self.root.title(f"Layer-4 事件流播放器 - {self.event_path.name}")
        self.reset_clock_anchor()
        self.update_ui(redraw_timeline=True)
        self.status_var.set(
            f"已从 {self.event_path} 加载 {len(self.timestamps):,} 个事件"
        )
        if self.autoplay:
            self.toggle_play()

    def _build_density_profile(self):
        duration = max(1.0, float(self.timestamps[-1] - self.timestamps[0]))
        self.density_counts, self.density_edges_us = np.histogram(
            self.timestamps - self.timestamps[0],
            bins=512,
            range=(0.0, duration),
        )

    def reset_clock_anchor(self):
        if len(self.timestamps):
            self.anchor_timestamp = float(self.playback_timestamp)
        self.anchor_wall_time = time.perf_counter()

    def toggle_play(self):
        if not len(self.timestamps):
            return
        if self.playing:
            self.playing = False
        else:
            first, last = self._active_bounds()
            start_t = float(self.timestamps[first])
            end_t = float(self.timestamps[last])
            if not (start_t <= self.playback_timestamp < end_t):
                self.set_index(first)
            self.playing = True
            self.reset_clock_anchor()
        self.play_button.configure(text="暂停" if self.playing else "播放")

    def restart(self):
        if not len(self.timestamps):
            return
        first, _ = self._active_bounds()
        self.set_index(first)

    def step_event(self, amount):
        if not len(self.timestamps):
            return
        self.playing = False
        self.play_button.configure(text="播放")
        first, last = self._active_bounds()
        self.set_index(int(np.clip(self.index + amount, first, last)))

    def change_speed(self, amount):
        nearest = min(
            range(len(SPEED_OPTIONS)),
            key=lambda index: abs(SPEED_OPTIONS[index] - self.speed),
        )
        nearest = int(np.clip(nearest + amount, 0, len(SPEED_OPTIONS) - 1))
        self.speed = SPEED_OPTIONS[nearest]
        self.speed_var.set(self._speed_text(self.speed))
        self.reset_clock_anchor()

    def on_speed_change(self, _event=None):
        text = self.speed_var.get().lower().replace("x", "")
        try:
            self.speed = float(text)
        except ValueError:
            return
        self.reset_clock_anchor()

    def on_display_change(self, _value=None):
        self._update_display_labels()
        if len(self.timestamps):
            self.draw_event_canvas()
            self.update_info()

    def _update_display_labels(self):
        self.trail_text.set(f"{self.trail_var.get():.0f} ms")
        self.fade_text.set(f"{self.fade_var.get():.0f} ms")
        self.gain_text.set(f"{self.gain_var.get():.2f}x")

    def _loop_bounds_from_entries(self, show_error=False):
        count = len(self.timestamps)
        if not count:
            return 0, 0
        try:
            first = int(self.loop_start_var.get()) - 1
            last = int(self.loop_end_var.get()) - 1
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "循环区间无效",
                    "循环起点和终点必须是整数事件序号。",
                    parent=self.root,
                )
                return None
            return 0, count - 1
        if not (0 <= first <= last < count):
            if show_error:
                messagebox.showerror(
                    "循环区间无效",
                    f"必须满足 1 <= 起点 <= 终点 <= {count}。",
                    parent=self.root,
                )
                return None
            return 0, count - 1
        return first, last

    def _active_bounds(self):
        if self.loop_enabled_var.get():
            return self._loop_bounds_from_entries(show_error=False)
        return 0, max(0, len(self.timestamps) - 1)

    def apply_loop_range(self):
        bounds = self._loop_bounds_from_entries(show_error=True)
        if bounds is None:
            return
        first, last = bounds
        if self.loop_enabled_var.get() and not (
            float(self.timestamps[first])
            <= self.playback_timestamp
            <= float(self.timestamps[last])
        ):
            self.set_index(first)
        self.draw_density_timeline()

    def on_loop_toggle(self):
        self.apply_loop_range()
        self.reset_clock_anchor()

    def jump_to_event(self, _event=None):
        if not len(self.timestamps):
            return
        try:
            number = int(self.event_number_var.get())
        except ValueError:
            self.event_number_var.set(str(self.index + 1))
            return
        self.playing = False
        self.play_button.configure(text="播放")
        self.set_index(int(np.clip(number - 1, 0, len(self.timestamps) - 1)))

    def on_progress(self, value):
        if self.updating_progress or not len(self.timestamps):
            return
        self.playing = False
        self.play_button.configure(text="播放")
        target = float(self.timestamps[0]) + float(value)
        self.set_playback_timestamp(target)

    def set_index(self, index):
        if not len(self.timestamps):
            return
        self.index = int(np.clip(index, 0, len(self.timestamps) - 1))
        self.playback_timestamp = float(self.timestamps[self.index])
        self.reset_clock_anchor()
        self.update_ui()

    def set_playback_timestamp(self, timestamp):
        """Seek the continuous playback clock without snapping to an event."""

        if not len(self.timestamps):
            return
        self.playback_timestamp = float(
            np.clip(timestamp, self.timestamps[0], self.timestamps[-1])
        )
        self.index = int(
            np.searchsorted(
                self.timestamps,
                self.playback_timestamp,
                side="right",
            )
            - 1
        )
        self.index = int(np.clip(self.index, 0, len(self.timestamps) - 1))
        self.reset_clock_anchor()
        self.update_ui()

    def tick(self):
        try:
            if self.playing and len(self.timestamps):
                first, last = self._active_bounds()
                elapsed = time.perf_counter() - self.anchor_wall_time
                target = self.anchor_timestamp + elapsed * 1_000_000.0 * self.speed

                if self.loop_enabled_var.get():
                    start_t = float(self.timestamps[first])
                    end_t = float(self.timestamps[last])
                    if target > end_t:
                        span = end_t - start_t
                        target = start_t if span <= 0 else start_t + (target - start_t) % span
                        self.anchor_timestamp = target
                        self.anchor_wall_time = time.perf_counter()
                elif target >= float(self.timestamps[last]):
                    target = float(self.timestamps[last])
                    self.playing = False
                    self.play_button.configure(text="播放")

                self.playback_timestamp = target
                new_index = int(np.searchsorted(self.timestamps, target, side="right") - 1)
                new_index = int(np.clip(new_index, first, last))
                self.index = new_index
                # Redraw every clock tick, even if no new event arrived.  This
                # keeps fading, playback time and progress moving through gaps.
                self.update_ui()
        finally:
            self.root.after(PLAYBACK_TICK_MS, self.tick)

    def update_ui(self, redraw_timeline=False):
        if not len(self.timestamps):
            return
        self.updating_progress = True
        try:
            relative_us = self.playback_timestamp - float(self.timestamps[0])
            self.progress_scale.set(relative_us)
        finally:
            self.updating_progress = False
        self.event_number_var.set(str(self.index + 1))
        duration_us = float(self.timestamps[-1] - self.timestamps[0])
        self.progress_text_var.set(
            f"时间 {format_duration(relative_us)} / {format_duration(duration_us)}"
            f"  ·  已到事件 {self.index + 1:,} / {len(self.timestamps):,}"
        )
        self.draw_event_canvas()
        if redraw_timeline:
            self.draw_density_timeline()
        else:
            self.update_timeline_cursor()
        self.update_info()

    def draw_event_canvas(self):
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)
        if not len(self.timestamps):
            canvas.create_text(
                width / 2,
                height / 2,
                text="请打开 Layer-4 CSV",
                fill="#aab7c4",
                font=("TkDefaultFont", 16),
            )
            return

        rgb, self.visible_count = build_event_rgb(
            self.timestamps,
            self.xs,
            self.ys,
            self.directions,
            self.index,
            self.playback_timestamp,
            self.trail_var.get() * 1000.0,
            self.fade_var.get() * 1000.0,
            self.gain_var.get(),
        )
        ppm = f"P6\n{IMAGE_SIZE} {IMAGE_SIZE}\n255\n".encode("ascii") + rgb.tobytes()
        if self.base_photo is None:
            self.base_photo = tk.PhotoImage(width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.base_photo.configure(data=ppm, format="PPM")

        scale = max(1, int(min(width - 24, height - 24) // IMAGE_SIZE))
        if scale == 1:
            self.image_photo = self.base_photo
        else:
            if self.scaled_photo is None or self.photo_scale != scale:
                self.scaled_photo = tk.PhotoImage(
                    width=IMAGE_SIZE * scale,
                    height=IMAGE_SIZE * scale,
                )
                self.photo_scale = scale
            self.scaled_photo.tk.call(
                str(self.scaled_photo),
                "copy",
                str(self.base_photo),
                "-zoom",
                scale,
                scale,
            )
            self.image_photo = self.scaled_photo

        image_width = IMAGE_SIZE * scale
        image_height = IMAGE_SIZE * scale
        left = (width - image_width) / 2.0
        top = (height - image_height) / 2.0
        canvas.create_image(left, top, image=self.image_photo, anchor="nw")

        for coordinate in (32, 64, 96):
            x = left + coordinate * scale
            y = top + coordinate * scale
            canvas.create_line(x, top, x, top + image_height, fill="#253242")
            canvas.create_line(left, y, left + image_width, y, fill="#253242")
        canvas.create_rectangle(
            left,
            top,
            left + image_width,
            top + image_height,
            outline="#607086",
        )

        x_value = int(self.xs[self.index])
        y_value = int(self.ys[self.index])
        direction = int(self.directions[self.index])
        marker_x = left + (x_value + 0.5) * scale
        marker_y = top + (y_value + 0.5) * scale
        radius = max(4, scale * 1.6)
        canvas.create_oval(
            marker_x - radius,
            marker_y - radius,
            marker_x + radius,
            marker_y + radius,
            outline="white",
            width=2,
        )
        dx, dy = ARROW_VECTORS[direction]
        length = max(10, scale * 3.0)
        canvas.create_line(
            marker_x,
            marker_y,
            marker_x + dx * length,
            marker_y + dy * length,
            fill=FLOW_DIRECTION_COLORS[direction],
            width=max(2, scale // 2),
            arrow=tk.LAST,
        )
        canvas.create_text(
            left + 5,
            top + 5,
            text="(0, 0)",
            fill="#aab7c4",
            anchor="nw",
        )
        canvas.create_text(
            left + image_width - 5,
            top + image_height - 5,
            text="(127, 127)",
            fill="#aab7c4",
            anchor="se",
        )

    def update_info(self):
        if not len(self.timestamps):
            return
        relative_us = int(self.playback_timestamp - self.timestamps[0])
        duration_us = int(self.timestamps[-1] - self.timestamps[0])
        direction = int(self.directions[self.index])
        self.info_var.set(
            f"文件：{self.event_path.name}\n"
            f"数据源：{self.source_shape}\n"
            f"事件总数：{len(self.timestamps):,}\n"
            f"总时长：{format_duration(duration_us)}\n\n"
            f"播放时间：{format_duration(relative_us)}\n"
            f"最近事件：{self.index + 1:,}\n"
            f"时间戳：{int(self.timestamps[self.index])}\n"
            f"位置：({int(self.xs[self.index])}, {int(self.ys[self.index])})\n"
            f"方向：c={direction}  {FLOW_DIRECTION_NAMES[direction]}\n"
            f"拖尾内事件：{self.visible_count:,}"
        )

    def draw_density_timeline(self):
        canvas = self.timeline
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 60)
        if not len(self.timestamps) or not len(self.density_counts):
            return
        pad_x = 10.0
        pad_top = 8.0
        pad_bottom = 18.0
        plot_width = width - 2 * pad_x
        plot_height = height - pad_top - pad_bottom

        if self.loop_enabled_var.get():
            first, last = self._active_bounds()
            duration = max(
                1.0,
                float(self.timestamps[-1] - self.timestamps[0]),
            )
            start_fraction = float(
                self.timestamps[first] - self.timestamps[0]
            ) / duration
            end_fraction = float(
                self.timestamps[last] - self.timestamps[0]
            ) / duration
            canvas.create_rectangle(
                pad_x + start_fraction * plot_width,
                pad_top,
                pad_x + end_fraction * plot_width,
                pad_top + plot_height,
                fill="#172b3d",
                outline="",
            )

        values = np.log1p(self.density_counts.astype(np.float64))
        maximum = max(float(values.max()), 1.0)
        points = [(pad_x, pad_top + plot_height)]
        for index, value in enumerate(values):
            x = pad_x + index / max(1, len(values) - 1) * plot_width
            y = pad_top + plot_height * (1.0 - value / maximum)
            points.append((x, y))
        points.append((pad_x + plot_width, pad_top + plot_height))
        canvas.create_polygon(points, fill="#176c9a", outline="")
        canvas.create_line(
            pad_x,
            pad_top + plot_height,
            pad_x + plot_width,
            pad_top + plot_height,
            fill="#607086",
        )
        canvas.create_text(
            pad_x,
            height - 3,
            text="0:00",
            anchor="sw",
            fill="#aab7c4",
        )
        canvas.create_text(
            pad_x + plot_width,
            height - 3,
            text=format_duration(self.timestamps[-1] - self.timestamps[0]),
            anchor="se",
            fill="#aab7c4",
        )
        self.timeline_cursor = canvas.create_line(
            pad_x,
            pad_top,
            pad_x,
            pad_top + plot_height,
            fill="#ffd60a",
            width=2,
        )
        self.update_timeline_cursor()

    def update_timeline_cursor(self):
        if not len(self.timestamps) or self.timeline_cursor is None:
            return
        width = max(self.timeline.winfo_width(), 200)
        height = max(self.timeline.winfo_height(), 60)
        pad_x = 10.0
        plot_width = width - 2 * pad_x
        duration = max(1.0, float(self.timestamps[-1] - self.timestamps[0]))
        fraction = float(self.playback_timestamp - self.timestamps[0]) / duration
        x = pad_x + fraction * plot_width
        self.timeline.coords(self.timeline_cursor, x, 8, x, height - 18)

    def on_timeline_click(self, event):
        if not len(self.timestamps):
            return
        width = max(self.timeline.winfo_width(), 200)
        fraction = np.clip((event.x - 10) / max(1, width - 20), 0.0, 1.0)
        target = self.timestamps[0] + fraction * (
            self.timestamps[-1] - self.timestamps[0]
        )
        self.playing = False
        self.play_button.configure(text="播放")
        self.set_playback_timestamp(target)

    def on_canvas_resize(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(40, self._finish_canvas_resize)

    def _finish_canvas_resize(self):
        self.resize_job = None
        self.draw_event_canvas()

    def on_timeline_resize(self, _event=None):
        if self.timeline_resize_job is not None:
            self.root.after_cancel(self.timeline_resize_job)
        self.timeline_resize_job = self.root.after(40, self._finish_timeline_resize)

    def _finish_timeline_resize(self):
        self.timeline_resize_job = None
        self.draw_density_timeline()

    def close(self):
        self.playing = False
        if self.high_resolution_timer:
            set_windows_timer_resolution(False)
            self.high_resolution_timer = False
        self.root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lightweight 128x128 / 64x64x8 Layer-4 event player"
    )
    parser.add_argument("event_csv", nargs="?", help="CSV to open at startup")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--trail-ms",
        "--persistence-ms",
        dest="trail_ms",
        type=float,
        default=160.0,
        help="visible source-time window in milliseconds",
    )
    parser.add_argument(
        "--fade-tau-ms",
        type=float,
        default=50.0,
        help="exponential fade time constant in milliseconds",
    )
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--autoplay", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = tk.Tk()
    EventStreamPlayer(
        root,
        event_path=args.event_csv,
        speed=args.speed,
        persistence_ms=args.trail_ms,
        fade_tau_ms=args.fade_tau_ms,
        gain=args.gain,
        autoplay=args.autoplay,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
