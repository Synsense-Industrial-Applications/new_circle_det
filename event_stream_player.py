"""事件流播放器：从 event_point_annotator.py 移除全部标注功能后的版本。

保留：
- 64x64x8 Layer-4 事件还原到 128x128；
- 四方向光流着色；
- 基于原始时间戳的播放、暂停、重新开始和变速；
- 自定义时间步长前进/后退；
- persistence 拖影与 gain；
- 时间轴、strength 最大值曲线、当前事件与当前位置标记。
- 固定有效圆内的 dir_score 点积和曲线与 strength 曲线。
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

import online_ring_scorer as scorer


IMAGE_SIZE = 128
PLAYBACK_TICK_MS = 16
SPEED_OPTIONS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
FLOW_DIRECTION_NAMES = ("↘", "↙", "↖", "↗")
FLOW_DIRECTION_COLORS = ("#ff453a", "#ffd60a", "#0a84ff", "#30d158")
FLOW_DIRECTION_RGB = np.asarray(
    [
        [255, 69, 58],
        [255, 214, 10],
        [10, 132, 255],
        [48, 209, 88],
    ],
    dtype=np.float32,
)
FEATURE_TO_DIRECTION = np.asarray(
    [0, 1, 1, 0, 2, 3, 3, 2],
    dtype=np.int8,
)
DIR_SCORE_WINDOW_US = 2_000
CHART_REFRESH_INTERVAL_S = 0.05
DIR_SCORE_CENTER_X = 68
DIR_SCORE_CENTER_Y = 83
DIR_SCORE_RADIUS = 50
DIR_SCORE_DIRECTION_VECTORS = np.asarray(
    [
        [1, -1],   # 右下
        [-1, -1],  # 左下
        [-1, 1],   # 左上
        [1, 1],    # 右上
    ],
    dtype=np.int16,
)


def set_windows_timer_resolution(enable):
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
    """加载 CSV，并按照原播放器的规则完成坐标与方向解码。"""
    path = Path(path)
    timestamps = []
    xs = []
    ys = []
    features = []

    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = {
            str(name).strip().lower(): name
            for name in (reader.fieldnames or [])
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
            raise ValueError("CSV 必须包含 timestamp、x、y 字段")

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamps.append(int(float(row[timestamp_field])))
                xs.append(int(float(row[x_field])))
                ys.append(int(float(row[y_field])))
                if feature_field is not None:
                    features.append(int(float(row[feature_field])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV 第 {row_number} 行数据无效：{exc}"
                ) from exc

    if not timestamps:
        raise ValueError("事件 CSV 为空")

    timestamps = np.asarray(timestamps, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int16)
    ys = np.asarray(ys, dtype=np.int16)
    flow_directions = np.zeros(len(timestamps), dtype=np.int8)
    source_shape = "128×128"

    if features:
        features = np.asarray(features, dtype=np.int16)
        if len(features) != len(timestamps):
            raise ValueError("feature 必须出现在每一行")
        if np.any((features < 0) | (features > 7)):
            raise ValueError("Layer-4 feature 必须在 0..7 范围内")
        flow_directions = FEATURE_TO_DIRECTION[features]

        if (
            np.all((xs >= 0) & (xs < 64))
            and np.all((ys >= 0) & (ys < 64))
        ):
            raw_x = xs.astype(np.int32)
            raw_y = ys.astype(np.int32)
            feature_mod4 = features.astype(np.int32) % 4
            xs = (
                raw_x * 2 + (feature_mod4 // 2) % 2
            ).astype(np.int16)
            ys = (
                raw_y * 2 + feature_mod4 % 2
            ).astype(np.int16)
            source_shape = "64×64×8 → 128×128（四方向光流）"

    invalid = (xs < 0) | (xs >= IMAGE_SIZE) | (ys < 0) | (ys >= IMAGE_SIZE)
    if np.any(invalid):
        bad = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"事件坐标必须在 0..127；索引 {bad} 为 "
            f"({int(xs[bad])}, {int(ys[bad])})"
        )

    order = np.argsort(timestamps, kind="stable")
    return (
        timestamps[order],
        xs[order],
        ys[order],
        flow_directions[order],
        source_shape,
    )


class EventStreamPlayer:
    def __init__(
        self,
        root,
        event_path=None,
        speed=1.0,
        persistence_ms=100.0,
        gain=1.0,
    ):
        self.root = root
        self.event_path = None
        self.timestamps = np.asarray([], dtype=np.int64)
        self.xs = np.asarray([], dtype=np.int16)
        self.ys = np.asarray([], dtype=np.int16)
        self.flow_directions = np.asarray([], dtype=np.int8)
        self.scores = np.empty((0, scorer.CLASS_COUNT), dtype=np.float32)
        self.strengths = np.asarray([], dtype=np.float32)
        self.dir_score_contributions = np.asarray([], dtype=np.int64)
        self.dir_scores = np.asarray([], dtype=np.int64)
        self.dir_score_times_us = np.asarray([], dtype=np.float64)
        self.first_timestamp = 0
        self.total_duration_us = 1
        self.index = 0
        self.cursor_relative_us = 0

        self.playing = False
        self.speed = float(speed)
        self.step_ms = 5.0
        self.persistence_us = max(1.0, float(persistence_ms) * 1_000.0)
        self.gain = max(0.01, float(gain))
        self.anchor_wall_time = time.perf_counter()
        self.anchor_event_timestamp = 0
        self.updating_seek = False

        self.event_base_photo = None
        self.event_scaled_photo = None
        self.event_photo = None
        self.event_photo_scale = None
        self.event_image_item = None
        self.current_marker_items = ()
        self.canvas_geometry = None
        self.event_view = (0.0, 0.0, 128.0)
        self.score_canvas_geometry = None
        self.score_bar_items = []
        self.score_value_items = []
        self.high_resolution_timer_enabled = set_windows_timer_resolution(True)

        self.density_bin_us = 5_000
        self.density_times_us = np.asarray([], dtype=np.float64)
        self.density_counts = np.asarray([], dtype=np.float64)
        self.density_plot = (50.0, 10.0, 100.0, 100.0)
        self.density_cursor_item = None
        self.density_cursor_text = None
        self.flow_plot = (52.0, 28.0, 100.0, 100.0)
        self.flow_cursor_item = None
        self.flow_cursor_text = None
        self.chart_half_window_ms = 10.0
        self.last_chart_redraw_wall = 0.0

        self.root.title("事件流播放器")
        self.root.geometry("1320x980")
        self.root.minsize(980, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._bind_keys()
        self.set_controls_enabled(False)
        self.root.after(PLAYBACK_TICK_MS, self.tick)

        if event_path is not None:
            self.root.after(100, lambda: self.load_file(Path(event_path)))
        else:
            self.root.after(100, self.choose_file)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            toolbar,
            text="打开 CSV",
            command=self.choose_file,
        ).pack(side=tk.LEFT)

        self.play_button = ttk.Button(
            toolbar,
            text="播放",
            command=self.toggle_play,
        )
        self.play_button.pack(side=tk.LEFT, padx=(10, 4))
        self.restart_button = ttk.Button(
            toolbar,
            text="重新开始",
            command=self.restart,
        )
        self.restart_button.pack(side=tk.LEFT, padx=4)

        self.step_back_button = ttk.Button(
            toolbar,
            text="◀",
            width=3,
            command=lambda: self.step_time(-1),
        )
        self.step_back_button.pack(side=tk.LEFT, padx=(10, 2))
        self.step_forward_button = ttk.Button(
            toolbar,
            text="▶",
            width=3,
            command=lambda: self.step_time(1),
        )
        self.step_forward_button.pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="步长").pack(side=tk.LEFT, padx=(5, 3))
        self.step_var = tk.DoubleVar(value=self.step_ms)
        self.step_spin = ttk.Spinbox(
            toolbar,
            from_=0.001,
            to=1_000_000,
            increment=1,
            width=7,
            textvariable=self.step_var,
            command=self.on_step_change,
        )
        self.step_spin.pack(side=tk.LEFT)
        self.step_spin.bind("<Return>", self.on_step_change)
        self.step_spin.bind("<FocusOut>", self.on_step_change)
        ttk.Label(toolbar, text="ms").pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(toolbar, text="速度").pack(side=tk.LEFT, padx=(14, 4))
        self.speed_var = tk.StringVar(value=f"{self.speed:g}x")
        self.speed_combo = ttk.Combobox(
            toolbar,
            width=7,
            state="readonly",
            textvariable=self.speed_var,
            values=[f"{value:g}x" for value in SPEED_OPTIONS],
        )
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        ttk.Label(toolbar, text="拖影").pack(side=tk.LEFT, padx=(14, 4))
        self.persistence_var = tk.DoubleVar(
            value=self.persistence_us / 1_000.0
        )
        self.persistence_spin = ttk.Spinbox(
            toolbar,
            from_=1,
            to=10_000,
            increment=10,
            width=8,
            textvariable=self.persistence_var,
            command=self.on_display_settings,
        )
        self.persistence_spin.pack(side=tk.LEFT)
        self.persistence_spin.bind("<Return>", self.on_display_settings)
        self.persistence_spin.bind("<FocusOut>", self.on_display_settings)
        ttk.Label(toolbar, text="ms").pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(toolbar, text="增益").pack(side=tk.LEFT, padx=(14, 4))
        self.gain_var = tk.DoubleVar(value=self.gain)
        self.show_region_overlay_var = tk.BooleanVar(value=True)
        self.chart_window_var = tk.DoubleVar(value=self.chart_half_window_ms)
        self.gain_spin = ttk.Spinbox(
            toolbar,
            from_=0.01,
            to=100,
            increment=0.1,
            width=7,
            textvariable=self.gain_var,
            command=self.on_display_settings,
        )
        self.gain_spin.pack(side=tk.LEFT)
        self.gain_spin.bind("<Return>", self.on_display_settings)
        self.gain_spin.bind("<FocusOut>", self.on_display_settings)

        self.region_overlay_check = ttk.Checkbutton(
            toolbar,
            text="显示 dir_score 区域",
            variable=self.show_region_overlay_var,
            command=self.on_region_overlay_toggle,
        )
        self.region_overlay_check.pack(side=tk.LEFT, padx=(14, 0))

        ttk.Label(toolbar, text="曲线窗口 ±").pack(
            side=tk.LEFT,
            padx=(14, 4),
        )
        self.chart_window_spin = ttk.Spinbox(
            toolbar,
            from_=0.1,
            to=1_000_000.0,
            increment=0.5,
            width=7,
            textvariable=self.chart_window_var,
            command=self.on_chart_window_change,
        )
        self.chart_window_spin.pack(side=tk.LEFT)
        self.chart_window_spin.bind(
            "<Return>",
            self.on_chart_window_change,
        )
        self.chart_window_spin.bind(
            "<FocusOut>",
            self.on_chart_window_change,
        )
        ttk.Label(toolbar, text="ms").pack(side=tk.LEFT, padx=(2, 0))

        content = ttk.Frame(self.root)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0, minsize=330)
        content.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            content,
            bg="#050505",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw_full_canvas())

        score_frame = ttk.LabelFrame(
            content,
            text="当前事件 get_score",
            padding=(6, 6),
        )
        score_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(8, 0),
        )
        score_frame.columnconfigure(0, weight=1)
        score_frame.rowconfigure(0, weight=1)
        self.score_canvas = tk.Canvas(
            score_frame,
            width=320,
            bg="#0b1220",
            highlightthickness=0,
        )
        self.score_canvas.grid(row=0, column=0, sticky="nsew")
        self.score_canvas.bind(
            "<Configure>",
            lambda _event: self.draw_score_chart(),
        )

        self.flow_frame = ttk.LabelFrame(
            self.root,
            text="dir_score｜圆心 (68, 83)，半径 50｜2ms 滑动窗口点积和",
            padding=(6, 4),
        )
        self.flow_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=(7, 0),
        )
        self.flow_frame.columnconfigure(0, weight=1)
        self.flow_canvas = tk.Canvas(
            self.flow_frame,
            height=235,
            bg="#0b1220",
            highlightthickness=0,
            cursor="hand2",
        )
        self.flow_canvas.grid(row=0, column=0, sticky="ew")
        self.flow_canvas.bind(
            "<Configure>",
            lambda _event: self.draw_flow_chart(),
        )
        self.flow_canvas.bind("<Button-1>", self.on_flow_click)

        density_frame = ttk.LabelFrame(
            self.root,
            text="strength 最大值随时间变化",
            padding=(6, 4),
        )
        density_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=10,
            pady=(7, 0),
        )
        density_frame.columnconfigure(0, weight=1)
        self.density_canvas = tk.Canvas(
            density_frame,
            height=145,
            bg="#0b1220",
            highlightthickness=0,
            cursor="hand2",
        )
        self.density_canvas.grid(row=0, column=0, sticky="ew")
        self.density_canvas.bind(
            "<Configure>",
            lambda _event: self.draw_density_chart(),
        )
        self.density_canvas.bind("<Button-1>", self.on_density_click)

        seek_frame = ttk.Frame(self.root, padding=(10, 7))
        seek_frame.grid(row=4, column=0, sticky="ew")
        seek_frame.columnconfigure(0, weight=1)
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            seek_frame,
            from_=0.0,
            to=1.0,
            variable=self.seek_var,
            command=self.on_seek,
        )
        self.seek_scale.grid(row=0, column=0, sticky="ew")
        self.time_var = tk.StringVar(value="00:00.000 / 00:00.000")
        ttk.Label(
            seek_frame,
            textvariable=self.time_var,
            width=24,
            anchor=tk.E,
        ).grid(row=0, column=1, padx=(10, 0))

        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        status_frame.grid(row=5, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="请选择事件 CSV")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        self.event_var = tk.StringVar(value="event: -")
        ttk.Label(
            status_frame,
            textvariable=self.event_var,
            anchor=tk.E,
        ).grid(row=0, column=1, padx=(12, 0))

        self.controls = [
            self.play_button,
            self.restart_button,
            self.step_back_button,
            self.step_forward_button,
            self.step_spin,
            self.speed_combo,
            self.persistence_spin,
            self.gain_spin,
            self.region_overlay_check,
            self.chart_window_spin,
            self.seek_scale,
        ]

    def _bind_keys(self):
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda event: self.on_step_key(event, -1))
        self.root.bind("<Right>", lambda event: self.on_step_key(event, 1))
        self.root.bind("<Home>", lambda _event: self.restart())

    @staticmethod
    def is_text_input(widget):
        return isinstance(
            widget,
            (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox),
        )

    def set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self.controls:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        if enabled:
            self.speed_combo.configure(state="readonly")

    @staticmethod
    def format_time(value_us):
        total_ms = max(0, int(value_us)) // 1_000
        minutes, remainder = divmod(total_ms, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def draw_score_chart(self):
        canvas = self.score_canvas
        canvas.delete("all")
        self.score_bar_items = []
        self.score_value_items = []
        width = max(canvas.winfo_width(), 260)
        height = max(canvas.winfo_height(), 360)
        self.score_canvas_geometry = (
            canvas.winfo_width(),
            canvas.winfo_height(),
        )

        left, right, top, bottom = 82.0, 46.0, 48.0, 24.0
        plot_width = max(1.0, width - left - right)
        plot_height = max(1.0, height - top - bottom)
        row_height = plot_height / scorer.CLASS_COUNT
        self.score_plot = (left, top, plot_width, row_height)

        self.score_prediction_item = canvas.create_text(
            width / 2,
            14,
            text="载入数据后显示 get_score",
            fill="#e2e8f0",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.score_strength_item = canvas.create_text(
            width / 2,
            32,
            text="strength: -",
            fill="#fbbf24",
            font=("Consolas", 9, "bold"),
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = left + fraction * plot_width
            canvas.create_line(
                x,
                top,
                x,
                top + plot_height,
                fill="#1e293b",
            )
            canvas.create_text(
                x,
                top + plot_height + 5,
                text=f"{fraction:g}",
                fill="#94a3b8",
                anchor=tk.N,
                font=("Segoe UI", 8),
            )

        for class_index, label in enumerate(scorer.CLASS_LABELS):
            center_y = top + (class_index + 0.5) * row_height
            canvas.create_text(
                left - 6,
                center_y,
                text=f"{class_index + 1:02d} {label}",
                fill="#cbd5e1",
                anchor=tk.E,
                font=("Microsoft YaHei UI", 8),
            )
            bar_item = canvas.create_rectangle(
                left,
                center_y - row_height * 0.28,
                left,
                center_y + row_height * 0.28,
                fill="#38bdf8",
                outline="",
            )
            value_item = canvas.create_text(
                left + 4,
                center_y,
                text="0.000000",
                fill="#e2e8f0",
                anchor=tk.W,
                font=("Consolas", 8),
            )
            self.score_bar_items.append(bar_item)
            self.score_value_items.append(value_item)
        self.update_score_chart()

    def update_score_chart(self):
        if (
            not len(self.timestamps)
            or self.scores.shape[0] != len(self.timestamps)
            or self.strengths.shape[0] != len(self.timestamps)
            or len(self.score_bar_items) != scorer.CLASS_COUNT
        ):
            return
        left, top, plot_width, row_height = self.score_plot
        values = self.scores[self.index]
        predicted_index = int(np.argmax(values))
        self.score_canvas.itemconfigure(
            self.score_prediction_item,
            text=(
                f"预测：{predicted_index + 1:02d} "
                f"{scorer.CLASS_LABELS[predicted_index]}  "
                f"{float(values[predicted_index]):.6f}"
            ),
        )
        self.score_canvas.itemconfigure(
            self.score_strength_item,
            text=f"strength: {float(self.strengths[self.index]):.6f}",
        )
        for class_index, value in enumerate(values):
            center_y = top + (class_index + 0.5) * row_height
            bar_end = left + float(np.clip(value, 0.0, 1.0)) * plot_width
            color = "#ff453a" if class_index == predicted_index else "#38bdf8"
            self.score_canvas.coords(
                self.score_bar_items[class_index],
                left,
                center_y - row_height * 0.28,
                bar_end,
                center_y + row_height * 0.28,
            )
            self.score_canvas.itemconfigure(
                self.score_bar_items[class_index],
                fill=color,
            )
            text_x = min(
                max(bar_end + 4, left + 4),
                left + plot_width + 4,
            )
            self.score_canvas.coords(
                self.score_value_items[class_index],
                text_x,
                center_y,
            )
            self.score_canvas.itemconfigure(
                self.score_value_items[class_index],
                text=f"{float(value):.6f}",
            )

    def precompute_scores(self):
        event_count = len(self.timestamps)
        self.scores = np.empty(
            (event_count, scorer.CLASS_COUNT),
            dtype=np.float32,
        )
        self.strengths = np.empty(event_count, dtype=np.float32)
        scorer.reset_Score()
        update_every = max(event_count // 200, 1)
        for event_index in range(event_count):
            score_info = scorer.get_Score(
                self.xs[event_index],
                self.ys[event_index],
                self.timestamps[event_index],
                return_info=True,
            )
            self.scores[event_index] = score_info["score"]
            self.strengths[event_index] = score_info["strength"]
            if (
                event_index % update_every == 0
                or event_index == event_count - 1
            ):
                self.status_var.set(
                    f"正在预计算 score 和 strength："
                    f"{event_index + 1:,}/{event_count:,}"
                )
                self.root.update_idletasks()
        self.precompute_dir_scores()

    def precompute_dir_scores(self):
        event_count = len(self.timestamps)
        if not event_count:
            self.dir_score_contributions = np.asarray([], dtype=np.int64)
            self.dir_scores = np.asarray([], dtype=np.int64)
            self.dir_score_times_us = np.asarray([], dtype=np.float64)
            return

        delta_x = self.xs.astype(np.int64) - DIR_SCORE_CENTER_X
        delta_y = DIR_SCORE_CENTER_Y - self.ys.astype(np.int64)
        active = (
            delta_x * delta_x + delta_y * delta_y
            <= DIR_SCORE_RADIUS * DIR_SCORE_RADIUS
        )
        direction_vectors = DIR_SCORE_DIRECTION_VECTORS[
            self.flow_directions.astype(np.intp, copy=False)
        ].astype(np.int64, copy=False)
        contributions = (
            delta_x * direction_vectors[:, 0]
            + delta_y * direction_vectors[:, 1]
        )
        contributions[~active] = 0
        self.dir_score_contributions = contributions

        window_left = np.searchsorted(
            self.timestamps,
            self.timestamps - DIR_SCORE_WINDOW_US,
            side="left",
        )
        prefix = np.empty(event_count + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(contributions, dtype=np.int64, out=prefix[1:])
        self.dir_scores = prefix[1:] - prefix[window_left]
        self.dir_score_times_us = (
            self.timestamps - self.first_timestamp
        ).astype(np.float64)
        self.status_var.set(
            f"dir_score 已预计算：{event_count:,} 个事件"
        )
        self.root.update_idletasks()

    @staticmethod
    def downsample_dir_score_curve(times, values, plot_width):
        target = max(2, int(plot_width))
        if len(values) <= target * 2:
            return times, values

        group_size = int(np.ceil(len(values) / target))
        group_count = int(np.ceil(len(values) / group_size))
        padded_size = group_count * group_size
        padded_min = np.full(padded_size, np.inf, dtype=np.float64)
        padded_max = np.full(padded_size, -np.inf, dtype=np.float64)
        padded_min[:len(values)] = values
        padded_max[:len(values)] = values
        minimum_offsets = np.argmin(
            padded_min.reshape(group_count, group_size),
            axis=1,
        )
        maximum_offsets = np.argmax(
            padded_max.reshape(group_count, group_size),
            axis=1,
        )
        starts = np.arange(group_count, dtype=np.int64) * group_size
        indices = np.sort(
            np.column_stack(
                [
                    starts + minimum_offsets,
                    starts + maximum_offsets,
                ]
            ),
            axis=1,
        ).ravel()
        indices = indices[indices < len(values)]
        if len(indices) > 1:
            indices = indices[
                np.concatenate(
                    [
                        np.asarray([True]),
                        np.diff(indices) != 0,
                    ]
                )
            ]
        return times[indices], values[indices]

    def draw_flow_chart(self):
        canvas = self.flow_canvas
        canvas.delete("all")
        self.flow_cursor_item = None
        self.flow_cursor_text = None
        width = max(canvas.winfo_width(), 240)
        height = max(canvas.winfo_height(), 200)
        left, right, top, bottom = 64.0, 14.0, 22.0, 26.0
        plot_width = max(1.0, width - left - right)
        plot_height = max(1.0, height - top - bottom)
        self.flow_plot = (left, top, plot_width, plot_height)

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + plot_height * fraction
            canvas.create_line(
                left,
                y,
                left + plot_width,
                y,
                fill="#475569" if fraction == 0.5 else "#1e293b",
                width=2 if fraction == 0.5 else 1,
                dash=() if fraction == 0.5 else (3, 4),
            )

        if not len(self.dir_scores):
            canvas.create_text(
                width / 2,
                height / 2,
                text="载入数据后显示 dir_score",
                fill="#64748b",
            )
            return

        half_window_us = self.chart_half_window_ms * 1_000.0
        window_start = self.cursor_relative_us - half_window_us
        window_end = self.cursor_relative_us + half_window_us
        start = int(
            np.searchsorted(
                self.dir_score_times_us,
                window_start,
                side="left",
            )
        )
        end = int(
            np.searchsorted(
                self.dir_score_times_us,
                window_end,
                side="right",
            )
        )
        times, values = self.downsample_dir_score_curve(
            self.dir_score_times_us[start:end],
            self.dir_scores[start:end],
            plot_width,
        )
        maximum = (
            max(float(np.max(np.abs(values))), 1.0)
            if len(values)
            else 1.0
        )
        x_values = left + (
            (times - window_start) / (2.0 * half_window_us)
        ) * plot_width
        y_values = (
            top
            + plot_height * 0.5
            - values / maximum * plot_height * 0.46
        )
        points = np.column_stack([x_values, y_values]).ravel().tolist()
        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill="#38bdf8",
                width=1,
                smooth=False,
            )
        elif len(points) == 2:
            point_x, point_y = points
            canvas.create_oval(
                point_x - 2,
                point_y - 2,
                point_x + 2,
                point_y + 2,
                fill="#38bdf8",
                outline="",
            )

        for label, label_y in (
            (f"+{maximum:g}", top),
            ("0", top + plot_height * 0.5),
            (f"-{maximum:g}", top + plot_height),
        ):
            canvas.create_text(
                left - 7,
                label_y,
                text=label,
                fill="#94a3b8",
                anchor=tk.E,
                font=("Segoe UI", 8),
            )
        canvas.create_text(
            left + 4,
            top + 3,
            text="2 ms 滑动窗口内的向量点积和",
            fill="#cbd5e1",
            anchor=tk.NW,
            font=("Microsoft YaHei UI", 9),
        )

        for fraction, label in (
            (0.0, f"-{self.chart_half_window_ms:g} ms"),
            (0.5, "0"),
            (1.0, f"+{self.chart_half_window_ms:g} ms"),
        ):
            tick_x = left + fraction * plot_width
            canvas.create_line(
                tick_x,
                top + plot_height,
                tick_x,
                top + plot_height + 4,
                fill="#64748b",
            )
            canvas.create_text(
                tick_x,
                top + plot_height + 6,
                text=label,
                fill="#94a3b8",
                anchor=tk.N,
                font=("Segoe UI", 8),
            )

        self.flow_cursor_item = canvas.create_line(
            left + plot_width * 0.5,
            top,
            left + plot_width * 0.5,
            top + plot_height,
            fill="#f43f5e",
            width=2,
        )
        self.flow_cursor_text = canvas.create_text(
            left + plot_width * 0.5 + 5,
            top + 3,
            text="",
            fill="#fda4af",
            anchor=tk.NW,
            font=("Segoe UI", 8, "bold"),
        )
        self.update_flow_cursor()

    def update_flow_cursor(self):
        if (
            not len(self.timestamps)
            or self.flow_cursor_item is None
            or self.flow_cursor_text is None
        ):
            return
        left, top, plot_width, plot_height = self.flow_plot
        x = left + 0.5 * plot_width
        self.flow_canvas.coords(
            self.flow_cursor_item,
            x,
            top,
            x,
            top + plot_height,
        )
        self.flow_canvas.coords(
            self.flow_cursor_text,
            x + 5,
            top + 3,
        )
        self.flow_canvas.itemconfigure(
            self.flow_cursor_text,
            text=self.format_time(self.cursor_relative_us),
            anchor=tk.NW,
        )

    def on_flow_click(self, event):
        if not len(self.timestamps):
            return
        left, _top, plot_width, _plot_height = self.flow_plot
        fraction = float(np.clip(
            (float(event.x) - left) / plot_width,
            0.0,
            1.0,
        ))
        offset_us = (
            (fraction - 0.5)
            * 2.0
            * self.chart_half_window_ms
            * 1_000.0
        )
        self.playing = False
        self.play_button.configure(text="播放")
        self.set_cursor_time(
            int(round(self.cursor_relative_us + offset_us))
        )

    def on_region_overlay_toggle(self):
        self.draw_region_overlay()

    def compute_density_profile(self):
        if not len(self.timestamps):
            self.density_times_us = np.asarray([], dtype=np.float64)
            self.density_counts = np.asarray([], dtype=np.float64)
            return
        relative = self.timestamps - self.first_timestamp
        bin_indices = (relative // self.density_bin_us).astype(np.int64)
        bin_count = int(bin_indices[-1]) + 1
        strength_maxima = np.full(
            bin_count,
            -np.inf,
            dtype=np.float64,
        )
        np.maximum.at(
            strength_maxima,
            bin_indices,
            self.strengths.astype(np.float64, copy=False),
        )
        strength_maxima[~np.isfinite(strength_maxima)] = 0.0
        self.density_counts = strength_maxima
        self.density_times_us = (
            np.arange(bin_count, dtype=np.float64) + 0.5
        ) * self.density_bin_us

    def density_curve_for_width(
        self,
        plot_width,
        times=None,
        counts=None,
    ):
        if times is None:
            times = self.density_times_us
        if counts is None:
            counts = self.density_counts
        if not len(counts):
            return times, counts
        target = max(2, int(plot_width))
        if len(counts) <= target * 2:
            return times, counts

        group_size = int(np.ceil(len(counts) / target))
        padded_size = int(np.ceil(len(counts) / group_size) * group_size)
        padded = np.full(padded_size, -np.inf, dtype=np.float64)
        padded[:len(counts)] = counts
        grouped = padded.reshape(-1, group_size)
        peak_offsets = np.argmax(grouped, axis=1)
        starts = np.arange(len(grouped), dtype=np.int64) * group_size
        peak_indices = np.minimum(starts + peak_offsets, len(counts) - 1)
        return times[peak_indices], counts[peak_indices]

    def draw_density_chart(self):
        canvas = self.density_canvas
        canvas.delete("all")
        self.density_cursor_item = None
        self.density_cursor_text = None
        width = max(canvas.winfo_width(), 160)
        height = max(canvas.winfo_height(), 120)
        left, right, top, bottom = 52.0, 14.0, 10.0, 25.0
        plot_width = max(1.0, width - left - right)
        plot_height = max(1.0, height - top - bottom)
        self.density_plot = (left, top, plot_width, plot_height)

        canvas.create_line(
            left,
            top + plot_height,
            left + plot_width,
            top + plot_height,
            fill="#475569",
        )
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = top + plot_height * (1.0 - fraction)
            canvas.create_line(
                left,
                y,
                left + plot_width,
                y,
                fill="#1e293b",
                dash=(3, 4),
            )

        if not len(self.density_counts):
            canvas.create_text(
                width / 2,
                height / 2,
                text="载入数据后显示 strength 最大值曲线",
                fill="#64748b",
            )
            return

        half_window_us = self.chart_half_window_ms * 1_000.0
        window_start = self.cursor_relative_us - half_window_us
        window_end = self.cursor_relative_us + half_window_us
        start = int(
            np.searchsorted(
                self.density_times_us,
                window_start,
                side="left",
            )
        )
        end = int(
            np.searchsorted(
                self.density_times_us,
                window_end,
                side="right",
            )
        )
        curve_times, curve_counts = self.density_curve_for_width(
            plot_width,
            self.density_times_us[start:end],
            self.density_counts[start:end],
        )
        maximum = (
            max(float(np.max(curve_counts)), 1.0)
            if len(curve_counts)
            else 1.0
        )
        x = left + (
            (curve_times - window_start) / (2.0 * half_window_us)
        ) * plot_width
        y = top + plot_height - curve_counts / maximum * plot_height
        points = np.column_stack([x, y]).ravel().tolist()
        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill="#38bdf8",
                width=1,
                smooth=False,
            )
        elif len(points) == 2:
            point_x, point_y = points
            canvas.create_oval(
                point_x - 2,
                point_y - 2,
                point_x + 2,
                point_y + 2,
                fill="#38bdf8",
                outline="",
            )

        canvas.create_text(
            left - 6,
            top,
            text=f"{maximum:.3f}",
            fill="#94a3b8",
            anchor=tk.E,
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            left - 6,
            top + plot_height,
            text="0",
            fill="#94a3b8",
            anchor=tk.E,
            font=("Segoe UI", 8),
        )
        for fraction, label in (
            (0.0, f"-{self.chart_half_window_ms:g} ms"),
            (0.5, "0"),
            (1.0, f"+{self.chart_half_window_ms:g} ms"),
        ):
            tick_x = left + fraction * plot_width
            canvas.create_line(
                tick_x,
                top + plot_height,
                tick_x,
                top + plot_height + 4,
                fill="#64748b",
            )
            canvas.create_text(
                tick_x,
                top + plot_height + 7,
                text=label,
                fill="#94a3b8",
                anchor=tk.N,
                font=("Segoe UI", 8),
            )

        canvas.create_text(
            left + 4,
            top + 3,
            text=(
                f"每 {self.density_bin_us / 1_000:g} ms "
                "strength 最大值"
            ),
            fill="#cbd5e1",
            anchor=tk.NW,
            font=("Microsoft YaHei UI", 9),
        )
        self.density_cursor_item = canvas.create_line(
            left + plot_width * 0.5,
            top,
            left + plot_width * 0.5,
            top + plot_height,
            fill="#f43f5e",
            width=2,
        )
        self.density_cursor_text = canvas.create_text(
            left + plot_width * 0.5 + 5,
            top + 4,
            text="",
            fill="#fda4af",
            anchor=tk.NW,
            font=("Segoe UI", 8, "bold"),
        )
        self.update_density_cursor()

    def update_density_cursor(self):
        if (
            not len(self.timestamps)
            or self.density_cursor_item is None
            or self.density_cursor_text is None
        ):
            return
        left, top, plot_width, plot_height = self.density_plot
        x = left + 0.5 * plot_width
        self.density_canvas.coords(
            self.density_cursor_item,
            x,
            top,
            x,
            top + plot_height,
        )
        self.density_canvas.coords(
            self.density_cursor_text,
            x + 5,
            top + 4,
        )
        self.density_canvas.itemconfigure(
            self.density_cursor_text,
            text=self.format_time(self.cursor_relative_us),
            anchor=tk.NW,
        )

    def on_density_click(self, event):
        if not len(self.timestamps):
            return
        left, _top, plot_width, _plot_height = self.density_plot
        fraction = float(np.clip(
            (float(event.x) - left) / plot_width,
            0.0,
            1.0,
        ))
        offset_us = (
            (fraction - 0.5)
            * 2.0
            * self.chart_half_window_ms
            * 1_000.0
        )
        self.playing = False
        self.play_button.configure(text="播放")
        self.set_cursor_time(
            int(round(self.cursor_relative_us + offset_us))
        )

    def choose_file(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择事件 CSV",
            initialdir=str(
                self.event_path.parent
                if self.event_path is not None
                else Path.cwd()
            ),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path):
        try:
            self.playing = False
            self.play_button.configure(text="播放")
            self.set_controls_enabled(False)
            self.status_var.set(f"正在载入 {path.name} …")
            self.root.update_idletasks()
            started = time.perf_counter()
            (
                timestamps,
                xs,
                ys,
                flow_directions,
                source_shape,
            ) = load_event_csv(path)

            self.event_path = path
            self.timestamps = timestamps
            self.xs = xs
            self.ys = ys
            self.flow_directions = flow_directions
            self.first_timestamp = int(timestamps[0])
            self.total_duration_us = max(
                1,
                int(timestamps[-1] - timestamps[0]),
            )
            self.precompute_scores()
            self.compute_density_profile()
            self.index = 0
            self.cursor_relative_us = 0
            self.seek_scale.configure(
                to=self.total_duration_us / 1_000_000.0
            )
            self.reset_clock_anchor()
            self.set_controls_enabled(True)
            self.update_ui(redraw=True, full_redraw=True)
            self.draw_score_chart()
            self.draw_density_chart()
            elapsed = time.perf_counter() - started
            self.root.title(f"事件流播放器 - {path.name}")
            self.status_var.set(
                f"{source_shape}｜{len(timestamps):,} 个事件｜"
                f"载入 {elapsed:.2f} s"
            )
        except Exception as exc:
            self.set_controls_enabled(len(self.timestamps) > 0)
            messagebox.showerror("载入失败", str(exc), parent=self.root)
            self.status_var.set(f"载入失败：{exc}")

    def reset_clock_anchor(self):
        self.anchor_wall_time = time.perf_counter()
        if len(self.timestamps):
            self.anchor_event_timestamp = (
                self.first_timestamp + int(self.cursor_relative_us)
            )

    def toggle_play(self):
        if not len(self.timestamps):
            return
        if self.playing:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        if self.index >= len(self.timestamps) - 1:
            self.set_index(0)
        self.playing = True
        self.play_button.configure(text="暂停")
        self.reset_clock_anchor()

    def restart(self):
        if not len(self.timestamps):
            return
        self.playing = False
        self.play_button.configure(text="播放")
        self.set_index(0)

    def step_time(self, amount):
        if not len(self.timestamps):
            return
        self.playing = False
        self.play_button.configure(text="播放")
        step_us = max(1, int(round(self.step_ms * 1_000.0)))
        self.set_cursor_time(
            self.cursor_relative_us + int(amount) * step_us
        )

    def on_step_key(self, event, amount):
        if self.is_text_input(event.widget):
            return None
        self.step_time(amount)
        return "break"

    def on_step_change(self, _event=None):
        try:
            value = float(self.step_var.get())
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            self.step_ms = float(np.clip(value, 0.001, 1_000_000.0))
            self.step_var.set(self.step_ms)
            self.status_var.set(f"时间步长已设为 {self.step_ms:g} ms")
        except (ValueError, tk.TclError):
            self.step_var.set(self.step_ms)

    def on_chart_window_change(self, _event=None):
        try:
            value = float(self.chart_window_var.get())
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            self.chart_half_window_ms = float(
                np.clip(value, 0.1, 1_000_000.0)
            )
            self.chart_window_var.set(self.chart_half_window_ms)
            if len(self.timestamps):
                self.update_local_charts(force=True)
            self.status_var.set(
                "两张曲线的显示范围已设为当前时刻前后各 "
                f"{self.chart_half_window_ms:g} ms"
            )
        except (ValueError, tk.TclError):
            self.chart_window_var.set(self.chart_half_window_ms)

    def set_cursor_time(self, relative_us):
        relative_us = int(
            np.clip(
                relative_us,
                0,
                self.total_duration_us,
            )
        )
        target_timestamp = self.first_timestamp + relative_us
        index = int(
            np.searchsorted(
                self.timestamps,
                target_timestamp,
                side="right",
            )
            - 1
        )
        self.index = int(np.clip(index, 0, len(self.timestamps) - 1))
        self.cursor_relative_us = relative_us
        self.reset_clock_anchor()
        self.update_ui(redraw=True)

    def set_index(self, index):
        self.index = int(np.clip(index, 0, len(self.timestamps) - 1))
        self.cursor_relative_us = int(
            self.timestamps[self.index] - self.first_timestamp
        )
        self.reset_clock_anchor()
        self.update_ui(redraw=True)

    def on_seek(self, value):
        if self.updating_seek or not len(self.timestamps):
            return
        self.playing = False
        self.play_button.configure(text="播放")
        relative_us = int(
            np.clip(
                float(value) * 1_000_000.0,
                0,
                self.total_duration_us,
            )
        )
        self.set_cursor_time(relative_us)

    def on_speed_change(self, _event=None):
        self.speed = float(self.speed_var.get().rstrip("x"))
        self.reset_clock_anchor()

    def on_display_settings(self, _event=None):
        try:
            self.persistence_us = max(
                1.0,
                float(self.persistence_var.get()) * 1_000.0,
            )
            self.gain = max(0.01, float(self.gain_var.get()))
            self.persistence_var.set(self.persistence_us / 1_000.0)
            self.gain_var.set(self.gain)
            if len(self.timestamps):
                self.update_ui(redraw=True, full_redraw=True)
        except (ValueError, tk.TclError):
            self.persistence_var.set(self.persistence_us / 1_000.0)
            self.gain_var.set(self.gain)

    def tick(self):
        if self.playing and len(self.timestamps):
            now = time.perf_counter()
            elapsed_us = (now - self.anchor_wall_time) * 1_000_000.0
            target_timestamp = (
                self.anchor_event_timestamp + elapsed_us * self.speed
            )
            new_index = int(
                np.searchsorted(
                    self.timestamps,
                    target_timestamp,
                    side="right",
                )
                - 1
            )
            new_index = min(
                max(new_index, self.index),
                len(self.timestamps) - 1,
            )
            self.cursor_relative_us = int(
                np.clip(
                    target_timestamp - self.first_timestamp,
                    0,
                    self.total_duration_us,
                )
            )

            if new_index != self.index:
                self.index = new_index
                self.update_ui(redraw=True)
            else:
                self.update_ui(redraw=False)

            if self.index >= len(self.timestamps) - 1:
                self.cursor_relative_us = self.total_duration_us
                self.playing = False
                self.play_button.configure(text="播放")
                self.update_ui(redraw=True)

        self.root.after(PLAYBACK_TICK_MS, self.tick)

    def update_ui(self, redraw=False, full_redraw=False):
        if not len(self.timestamps):
            return
        if redraw:
            if full_redraw:
                self.draw_full_canvas()
            else:
                self.draw_playback_frame()
            self.update_score_chart()

        self.updating_seek = True
        self.seek_var.set(self.cursor_relative_us / 1_000_000.0)
        self.updating_seek = False
        self.time_var.set(
            f"{self.format_time(self.cursor_relative_us)} / "
            f"{self.format_time(self.total_duration_us)}"
        )
        self.event_var.set(
            f"event: {self.index + 1:,}/{len(self.timestamps):,}  "
            f"t={int(self.timestamps[self.index])} μs  "
            f"光流={FLOW_DIRECTION_NAMES[int(self.flow_directions[self.index])]}"
        )
        self.update_local_charts(force=not self.playing)

    def update_local_charts(self, force=False):
        now = time.perf_counter()
        if (
            not force
            and now - self.last_chart_redraw_wall
            < CHART_REFRESH_INTERVAL_S
        ):
            self.update_flow_cursor()
            self.update_density_cursor()
            return
        self.last_chart_redraw_wall = now
        self.draw_flow_chart()
        self.draw_density_chart()

    def build_rgb(self):
        current_timestamp = int(self.timestamps[self.index])
        cutoff = current_timestamp - 6.0 * self.persistence_us
        start = int(
            np.searchsorted(
                self.timestamps,
                cutoff,
                side="left",
            )
        )
        recent = slice(start, self.index + 1)
        ages = current_timestamp - self.timestamps[recent]
        values = np.exp(-ages / self.persistence_us).astype(np.float32)
        flat = (
            self.ys[recent].astype(np.int32) * IMAGE_SIZE
            + self.xs[recent].astype(np.int32)
        )
        recent_directions = self.flow_directions[recent]
        heat = np.zeros(
            (4, IMAGE_SIZE * IMAGE_SIZE),
            dtype=np.float32,
        )
        for direction in range(4):
            mask = recent_directions == direction
            if np.any(mask):
                np.add.at(
                    heat[direction],
                    flat[mask],
                    values[mask],
                )

        normalized = 1.0 - np.exp(-self.gain * heat)
        rgb = np.einsum(
            "dh,dc->hc",
            normalized,
            FLOW_DIRECTION_RGB,
            optimize=True,
        )
        return np.clip(
            rgb.reshape(IMAGE_SIZE, IMAGE_SIZE, 3),
            0,
            255,
        ).astype(np.uint8)

    def render_photo(self, scale):
        rgb = self.build_rgb()
        ppm = (
            f"P6\n{IMAGE_SIZE} {IMAGE_SIZE}\n255\n".encode("ascii")
            + rgb.tobytes()
        )
        if self.event_base_photo is None:
            self.event_base_photo = tk.PhotoImage(
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
            )
        self.event_base_photo.configure(data=ppm, format="PPM")

        if scale <= 1:
            self.event_photo = self.event_base_photo
            return
        if (
            self.event_scaled_photo is None
            or self.event_photo_scale != scale
        ):
            self.event_scaled_photo = tk.PhotoImage(
                width=IMAGE_SIZE * scale,
                height=IMAGE_SIZE * scale,
            )
            self.event_photo_scale = scale
        self.event_scaled_photo.tk.call(
            str(self.event_scaled_photo),
            "copy",
            str(self.event_base_photo),
            "-zoom",
            scale,
            scale,
        )
        self.event_photo = self.event_scaled_photo

    def draw_full_canvas(self):
        canvas = self.canvas
        canvas.delete("all")
        self.event_image_item = None
        self.current_marker_items = ()
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)
        self.canvas_geometry = (
            canvas.winfo_width(),
            canvas.winfo_height(),
        )
        if not len(self.timestamps):
            canvas.create_text(
                width / 2,
                height / 2,
                text="请选择事件 CSV",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 18, "bold"),
            )
            return

        margin = 62
        usable = min(width - 2 * margin, height - 2 * margin)
        scale = max(1, int(usable // IMAGE_SIZE))
        side = IMAGE_SIZE * scale
        left = (width - side) / 2
        top = (height - side) / 2
        self.event_view = (left, top, side)
        self.render_photo(scale)
        self.event_image_item = canvas.create_image(
            left,
            top,
            image=self.event_photo,
            anchor=tk.NW,
        )
        canvas.create_rectangle(
            left,
            top,
            left + side,
            top + side,
            outline="#64748b",
        )
        self.draw_region_overlay()
        self.current_marker_items = (
            canvas.create_line(0, 0, 0, 0, fill="#ffffff"),
            canvas.create_line(0, 0, 0, 0, fill="#ffffff"),
        )
        self.update_marker()
        canvas.create_text(
            width / 2,
            22,
            text=(
                f"128×128 事件画面    "
                f"拖影：{self.persistence_us / 1_000.0:g} ms"
            ),
            fill="#e2e8f0",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        legend_y = 46
        legend_width = 104
        legend_start = width / 2 - legend_width * 2
        for direction, (name, color) in enumerate(
            zip(FLOW_DIRECTION_NAMES, FLOW_DIRECTION_COLORS)
        ):
            x = legend_start + direction * legend_width
            canvas.create_rectangle(
                x,
                legend_y - 6,
                x + 12,
                legend_y + 6,
                fill=color,
                outline="",
            )
            canvas.create_text(
                x + 18,
                legend_y,
                text=name,
                fill="#cbd5e1",
                anchor=tk.W,
                font=("Segoe UI Symbol", 14, "bold"),
            )

    def draw_region_overlay(self):
        canvas = self.canvas
        canvas.delete("region_overlay")
        if (
            not len(self.timestamps)
            or not self.show_region_overlay_var.get()
        ):
            return

        left, top, side = self.event_view
        center_x = float(DIR_SCORE_CENTER_X)
        center_y = float(DIR_SCORE_CENTER_Y)
        radius = float(DIR_SCORE_RADIUS)

        def to_canvas(point_x, point_y):
            return (
                left + point_x / IMAGE_SIZE * side,
                top + point_y / IMAGE_SIZE * side,
            )

        def draw_clipped_circle(circle_radius, color, width):
            angles = np.linspace(0.0, 2.0 * np.pi, 257)
            circle_x = center_x + circle_radius * np.cos(angles)
            circle_y = center_y + circle_radius * np.sin(angles)
            visible = (
                (circle_x >= 0.0)
                & (circle_x <= IMAGE_SIZE)
                & (circle_y >= 0.0)
                & (circle_y <= IMAGE_SIZE)
            )
            segments = []
            segment = []
            for point_x, point_y, is_visible in zip(
                circle_x,
                circle_y,
                visible,
            ):
                if is_visible:
                    segment.append(to_canvas(point_x, point_y))
                elif segment:
                    segments.append(segment)
                    segment = []
            if segment:
                segments.append(segment)
            if (
                len(segments) > 1
                and visible[0]
                and visible[-1]
            ):
                segments[0] = segments[-1] + segments[0]
                segments.pop()
            for points in segments:
                if len(points) < 2:
                    continue
                canvas.create_line(
                    *[
                        coordinate
                        for point in points
                        for coordinate in point
                    ],
                    fill=color,
                    width=width,
                    smooth=True,
                    tags=("region_overlay",),
                )

        inner_radius = 0.0
        draw_clipped_circle(radius, "#f8fafc", 2)

        def draw_horizontal(start_x, end_x):
            if not 0.0 <= center_y <= IMAGE_SIZE:
                return
            start_x = max(0.0, start_x)
            end_x = min(float(IMAGE_SIZE), end_x)
            if end_x <= start_x:
                return
            x0, line_y = to_canvas(start_x, center_y)
            x1, _ = to_canvas(end_x, center_y)
            canvas.create_line(
                x0,
                line_y,
                x1,
                line_y,
                fill="#fbbf24",
                width=2,
                tags=("region_overlay",),
            )

        def draw_vertical(start_y, end_y):
            if not 0.0 <= center_x <= IMAGE_SIZE:
                return
            start_y = max(0.0, start_y)
            end_y = min(float(IMAGE_SIZE), end_y)
            if end_y <= start_y:
                return
            line_x, y0 = to_canvas(center_x, start_y)
            _, y1 = to_canvas(center_x, end_y)
            canvas.create_line(
                line_x,
                y0,
                line_x,
                y1,
                fill="#fbbf24",
                width=2,
                tags=("region_overlay",),
            )

        draw_horizontal(center_x - radius, center_x - inner_radius)
        draw_horizontal(center_x + inner_radius, center_x + radius)
        draw_vertical(center_y - radius, center_y - inner_radius)
        draw_vertical(center_y + inner_radius, center_y + radius)

        canvas_x, canvas_y = to_canvas(center_x, center_y)
        canvas.create_oval(
            canvas_x - 3,
            canvas_y - 3,
            canvas_x + 3,
            canvas_y + 3,
            fill="#fbbf24",
            outline="",
            tags=("region_overlay",),
        )
        canvas.create_text(
            canvas_x + 7,
            canvas_y - 7,
            text="(68, 83)  r=50",
            fill="#ffffff",
            anchor=tk.SW,
            font=("Consolas", 9, "bold"),
            tags=("region_overlay",),
        )

        for marker_item in self.current_marker_items:
            canvas.tag_raise(marker_item)

    def draw_playback_frame(self):
        geometry = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if (
            self.event_image_item is None
            or len(self.current_marker_items) != 2
            or self.canvas_geometry != geometry
        ):
            self.draw_full_canvas()
            return
        _left, _top, side = self.event_view
        scale = max(1, int(round(side / IMAGE_SIZE)))
        self.render_photo(scale)
        self.canvas.itemconfigure(
            self.event_image_item,
            image=self.event_photo,
        )
        self.update_marker()

    def update_marker(self):
        if not len(self.timestamps) or len(self.current_marker_items) != 2:
            return
        left, top, side = self.event_view
        x = left + (float(self.xs[self.index]) + 0.5) / IMAGE_SIZE * side
        y = top + (float(self.ys[self.index]) + 0.5) / IMAGE_SIZE * side
        horizontal, vertical = self.current_marker_items
        self.canvas.coords(horizontal, x - 5, y, x + 5, y)
        self.canvas.coords(vertical, x, y - 5, x, y + 5)

    def close(self):
        self.playing = False
        if self.high_resolution_timer_enabled:
            set_windows_timer_resolution(False)
            self.high_resolution_timer_enabled = False
        self.root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="播放 128×128 或 64×64×8 事件流 CSV"
    )
    parser.add_argument(
        "event_csv",
        nargs="?",
        help="启动后自动加载的事件 CSV",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--persistence-ms", type=float, default=100.0)
    parser.add_argument("--gain", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = tk.Tk()
    EventStreamPlayer(
        root,
        event_path=args.event_csv,
        speed=args.speed,
        persistence_ms=args.persistence_ms,
        gain=args.gain,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
