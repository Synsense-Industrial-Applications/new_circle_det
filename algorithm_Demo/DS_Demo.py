"""Real-time separable D/S centre-vote demo for the current Layer-4 SNN.

The hardware/network configuration is imported from ``Demo.py`` so there is
only one SNN definition to maintain.  The last 200 decoded Layer-4 events form
a sliding window.  A new D/S result is computed after every 6 new events.

Only the two one-dimensional histogram peaks are used here.  Radius fitting,
circle coverage, tracking and the existing confidence pipeline are purposely
not part of this demo.

Run:
    python DS_Demo.py
    python DS_Demo.py --peak-threshold 0.15 --display-fps 30

Keys:
    [ / ]       decrease / increase the peak threshold by 0.02
    Q or Esc     close the demo

The last centre that passes the threshold remains visible for 3 seconds.  The
fixed cue-head reference defaults to (68, 83) and can be changed with
``--cue-x`` and ``--cue-y``.

The UI uses Tk (Python standard library) and runs in the main thread.  Hardware
reading and D/S calculation run in a worker thread.  A one-element queue keeps
only the newest snapshot, so a burst of events cannot build a rendering backlog.
The implementation contains no Windows-only input or display API and runs on
Linux when Tk, NumPy, samna and the Speck2f runtime are installed.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import traceback

import numpy as np


IMAGE_SIZE = 128
WINDOW_EVENTS = 200
UPDATE_STEP_EVENTS = 6
READ_BATCH_SIZE = 512
READ_TIMEOUT_MS = 10

D_FAMILY = frozenset((0, 3, 4, 7))
S_FAMILY = frozenset((1, 2, 5, 6))
FEATURE_TO_DIRECTION = np.asarray((0, 1, 1, 0, 2, 3, 3, 2), dtype=np.int8)
DIRECTION_RGB = np.asarray(
    (
        (255.0, 69.0, 58.0),
        (255.0, 214.0, 10.0),
        (10.0, 132.0, 255.0),
        (48.0, 209.0, 88.0),
    ),
    dtype=np.float32,
)

# A short symmetric kernel reduces 1-pixel address quantisation without SciPy.
SMOOTH_KERNEL = np.asarray((1, 2, 3, 4, 3, 2, 1), dtype=np.float32) / 16.0
PEAK_SUPPORT_HALF_WIDTH = 3
SECOND_PEAK_EXCLUSION = 8
MIN_FAMILY_EVENTS = 8
THRESHOLD_STEP = 0.02
PREDICTION_HOLD_SEC = 3.0
DEFAULT_CUE_HEAD_X = 68.0
DEFAULT_CUE_HEAD_Y = 83.0


@dataclass(frozen=True)
class PeakResult:
    value: float
    second_value: float
    support: float
    prominence: float
    score: float


@dataclass(frozen=True)
class DSResult:
    cx: float
    cy: float
    d_peak: PeakResult
    s_peak: PeakResult
    score: float
    d_event_count: int
    s_event_count: int
    inside_view: bool
    enough_events: bool


@dataclass(frozen=True)
class DSSnapshot:
    x: np.ndarray
    y: np.ndarray
    feature: np.ndarray
    timestamp: np.ndarray
    d_histogram: np.ndarray
    s_histogram: np.ndarray
    result: DSResult
    source_event_count: int
    update_count: int
    published_wall_time: float


class ThresholdState:
    """Thread-safe, live-adjustable display/detection threshold."""

    def __init__(self, value: float):
        self._lock = threading.Lock()
        self._value = float(np.clip(value, 0.0, 1.0))

    def get(self) -> float:
        with self._lock:
            return self._value

    def change(self, amount: float) -> float:
        with self._lock:
            self._value = float(np.clip(self._value + amount, 0.0, 1.0))
            return self._value


class SharedRuntimeStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = "starting"
        self.layer4_rate = 0.0
        self.update_rate = 0.0
        self.error = ""

    def update(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def read(self):
        with self._lock:
            return self.state, self.layer4_rate, self.update_rate, self.error


def decode_layer4_address(x64: int, y64: int, feature: int):
    """Decode the current 64x64x8 Layer-4 address to one 128x128 point."""

    feature = int(feature)
    if not 0 <= feature <= 7:
        raise ValueError(f"Layer-4 feature must be in 0..7, got {feature}")
    feature_mod4 = feature % 4
    x128 = 2 * int(x64) + (feature_mod4 // 2) % 2
    y128 = 2 * int(y64) + feature_mod4 % 2
    if not (0 <= x128 < IMAGE_SIZE and 0 <= y128 < IMAGE_SIZE):
        raise ValueError(f"decoded point is outside 128x128: ({x128}, {y128})")
    return x128, y128


def _peak_from_histogram(histogram: np.ndarray, value_offset: int) -> PeakResult:
    """Return peak location and a bounded measure of peak obviousness.

    ``support`` is the fraction of this direction family's events within
    +/-3 bins of the primary peak.  ``prominence`` compares the smoothed
    primary peak with the strongest competing peak outside +/-8 bins.
    The final per-family score is support with a moderate ambiguity penalty.
    """

    total = float(histogram.sum())
    if total <= 0.0:
        return PeakResult(np.nan, np.nan, 0.0, 0.0, 0.0)

    smooth = np.convolve(histogram, SMOOTH_KERNEL, mode="same")
    primary_index = int(np.argmax(smooth))
    primary_height = float(smooth[primary_index])

    support_lo = max(0, primary_index - PEAK_SUPPORT_HALF_WIDTH)
    support_hi = min(len(histogram), primary_index + PEAK_SUPPORT_HALF_WIDTH + 1)
    local_histogram = histogram[support_lo:support_hi].astype(np.float64)
    local_values = np.arange(support_lo, support_hi, dtype=np.float64) + value_offset
    if local_histogram.sum() > 0.0:
        primary_value = float(np.average(local_values, weights=local_histogram))
    else:
        primary_value = float(primary_index + value_offset)
    support = float(local_histogram.sum() / total)

    competitors = smooth.copy()
    exclude_lo = max(0, primary_index - SECOND_PEAK_EXCLUSION)
    exclude_hi = min(len(histogram), primary_index + SECOND_PEAK_EXCLUSION + 1)
    competitors[exclude_lo:exclude_hi] = 0.0
    second_index = int(np.argmax(competitors))
    second_height = float(competitors[second_index])
    second_value = float(second_index + value_offset)
    prominence = float(
        np.clip((primary_height - second_height) / max(primary_height, 1e-9), 0.0, 1.0)
    )
    score = float(support * (0.5 + 0.5 * prominence))
    return PeakResult(primary_value, second_value, support, prominence, score)


class DSWindow:
    """A 200-event sliding window recalculated every 6 input events."""

    def __init__(self):
        self._events = deque(maxlen=WINDOW_EVENTS)
        self.source_event_count = 0
        self.update_count = 0
        self._events_since_update = 0

    def push_raw(self, x64, y64, feature, timestamp):
        x, y = decode_layer4_address(x64, y64, feature)
        self._events.append((x, y, int(feature), int(timestamp)))
        self.source_event_count += 1
        self._events_since_update += 1
        if len(self._events) < WINDOW_EVENTS:
            return None
        if self.update_count and self._events_since_update < UPDATE_STEP_EVENTS:
            return None
        self._events_since_update = 0
        self.update_count += 1
        return self._calculate()

    def _calculate(self):
        values = np.asarray(self._events, dtype=np.int64)
        x = values[:, 0].astype(np.int16)
        y = values[:, 1].astype(np.int16)
        feature = values[:, 2].astype(np.int8)
        timestamp = values[:, 3].astype(np.int64)

        d_mask = np.isin(feature, tuple(D_FAMILY))
        s_mask = ~d_mask
        d_values = y[d_mask].astype(np.int16) - x[d_mask].astype(np.int16)
        s_values = x[s_mask].astype(np.int16) + y[s_mask].astype(np.int16)
        d_histogram = np.bincount(d_values + 127, minlength=255).astype(np.float32)
        s_histogram = np.bincount(s_values, minlength=255).astype(np.float32)

        d_peak = _peak_from_histogram(d_histogram, -127)
        s_peak = _peak_from_histogram(s_histogram, 0)
        if np.isfinite(d_peak.value + s_peak.value):
            cx = (s_peak.value - d_peak.value) / 2.0
            cy = (s_peak.value + d_peak.value) / 2.0
        else:
            cx = cy = np.nan
        d_count = int(d_mask.sum())
        s_count = int(s_mask.sum())
        result = DSResult(
            cx=float(cx),
            cy=float(cy),
            d_peak=d_peak,
            s_peak=s_peak,
            score=float(min(d_peak.score, s_peak.score)),
            d_event_count=d_count,
            s_event_count=s_count,
            inside_view=bool(0.0 <= cx < IMAGE_SIZE and 0.0 <= cy < IMAGE_SIZE),
            enough_events=bool(
                d_count >= MIN_FAMILY_EVENTS and s_count >= MIN_FAMILY_EVENTS
            ),
        )
        return DSSnapshot(
            x=x,
            y=y,
            feature=feature,
            timestamp=timestamp,
            d_histogram=d_histogram,
            s_histogram=s_histogram,
            result=result,
            source_event_count=self.source_event_count,
            update_count=self.update_count,
            published_wall_time=time.monotonic(),
        )


def publish_latest(output_queue: queue.Queue, snapshot: DSSnapshot):
    """Publish without blocking; discard an obsolete unrendered snapshot."""

    try:
        output_queue.put_nowait(snapshot)
        return
    except queue.Full:
        pass
    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        output_queue.put_nowait(snapshot)
    except queue.Full:
        pass


class HardwareWorker(threading.Thread):
    """Own the board event loop; never waits for the GUI renderer."""

    def __init__(self, output_queue, stop_event, runtime_status):
        super().__init__(name="ds-hardware", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.runtime_status = runtime_status

    def run(self):
        input_graph = None
        device_input_route = None
        try:
            # Delay hardware imports so --help and offline algorithm tests work
            # on development machines without the Speck2f runtime installed.
            import samna
            from Demo import (
                config,
                configure_cnn_pipeline,
                layer_4,
                open_speck2f_dev_kit,
            )

            self.runtime_status.update(state="configuring current SNN")
            configure_cnn_pipeline()
            dev_kit = open_speck2f_dev_kit()

            input_graph = samna.graph.EventFilterGraph()
            input_buffer = samna.BasicSourceNode_speck2f_event_input_event()
            input_graph.sequential([input_buffer, dev_kit.get_model_sink_node()])
            input_graph.start()
            dev_kit.get_model().apply_configuration(config)

            # Keep this source route alive until the graph is stopped.
            device_input_route = samna.graph.source_to(dev_kit.get_model_sink_node())
            event_buffer = samna.graph.sink_from(dev_kit.get_model_source_node())

            io_module = dev_kit.get_io_module()
            io_module.set_slow_clk_rate(32)
            io_module.set_slow_clk(True)
            io_module.set_in_out_interface_clk_rate(1_000_000)
            dev_kit.get_power_module().set_vdd_io(3.3)
            stopwatch = dev_kit.get_stop_watch()
            stopwatch.reset()
            stopwatch.start()

            event_buffer.get_events()
            detector = DSWindow()
            self.runtime_status.update(state="running")
            rate_started = time.monotonic()
            previous_events = 0
            previous_updates = 0

            while not self.stop_event.is_set():
                events = event_buffer.get_n_events(
                    n=READ_BATCH_SIZE,
                    timeout=READ_TIMEOUT_MS,
                )
                for event in events:
                    if getattr(event, "layer", None) != layer_4:
                        continue
                    try:
                        snapshot = detector.push_raw(
                            getattr(event, "x"),
                            getattr(event, "y"),
                            getattr(event, "feature"),
                            getattr(event, "timestamp"),
                        )
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if snapshot is not None:
                        publish_latest(self.output_queue, snapshot)

                now = time.monotonic()
                elapsed = now - rate_started
                if elapsed >= 1.0:
                    self.runtime_status.update(
                        layer4_rate=(detector.source_event_count - previous_events) / elapsed,
                        update_rate=(detector.update_count - previous_updates) / elapsed,
                    )
                    previous_events = detector.source_event_count
                    previous_updates = detector.update_count
                    rate_started = now

        except BaseException as error:
            self.runtime_status.update(
                state="error",
                error=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            )
        finally:
            if input_graph is not None:
                try:
                    input_graph.stop()
                except Exception:
                    pass
            # Intentional lifetime anchor for the device route.
            _ = device_input_route
            if not self.runtime_status.read()[3]:
                self.runtime_status.update(state="stopped")


class DSViewer:
    SCALE = 4
    EVENT_SIDE = IMAGE_SIZE * SCALE
    HISTOGRAM_WIDTH = 470
    HISTOGRAM_HEIGHT = 235

    def __init__(
        self,
        root,
        output_queue,
        stop_event,
        threshold_state,
        runtime_status,
        display_fps=30.0,
        fade_tau_ms=80.0,
        cue_head_x=DEFAULT_CUE_HEAD_X,
        cue_head_y=DEFAULT_CUE_HEAD_Y,
    ):
        self.root = root
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.threshold_state = threshold_state
        self.runtime_status = runtime_status
        self.frame_interval_ms = max(10, int(round(1000.0 / max(display_fps, 1.0))))
        self.fade_tau_us = max(float(fade_tau_ms) * 1000.0, 1.0)
        self.snapshot = None
        self.last_histogram_update = -1
        self.last_prediction = None
        self.last_prediction_wall_time = 0.0
        self.base_photo = tk.PhotoImage(width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.scaled_photo = tk.PhotoImage(width=self.EVENT_SIDE, height=self.EVENT_SIDE)
        self.image_item = None

        self.threshold_text = tk.StringVar()
        self.result_text = tk.StringVar(value="Waiting for 200 Layer-4 events...")
        self.runtime_text = tk.StringVar(value="Starting hardware...")

        root.title("Layer-4 D/S real-time centre demo")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("q", lambda _event: self.close())
        root.bind("[", lambda _event: self.change_threshold(-THRESHOLD_STEP))
        root.bind("]", lambda _event: self.change_threshold(THRESHOLD_STEP))

        outer = ttk.Frame(root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="n", padx=(0, 12))
        ttk.Label(left, text="Layer-4 decoded events (128 x 128)").pack(anchor="w")
        self.event_canvas = tk.Canvas(
            left,
            width=self.EVENT_SIDE,
            height=self.EVENT_SIDE,
            background="#03060a",
            highlightthickness=0,
        )
        self.event_canvas.pack()
        self.image_item = self.event_canvas.create_image(
            0, 0, image=self.scaled_photo, anchor="nw"
        )
        for coordinate in (32, 64, 96):
            location = coordinate * self.SCALE
            self.event_canvas.create_line(
                location, 0, location, self.EVENT_SIDE, fill="#263240"
            )
            self.event_canvas.create_line(
                0, location, self.EVENT_SIDE, location, fill="#263240"
            )
        cue_x = (float(cue_head_x) + 0.5) * self.SCALE
        cue_y = (float(cue_head_y) + 0.5) * self.SCALE
        cue_radius = 7.0
        self.cue_circle = self.event_canvas.create_oval(
            cue_x - cue_radius,
            cue_y - cue_radius,
            cue_x + cue_radius,
            cue_y + cue_radius,
            outline="#ff9f0a",
            width=2,
        )
        self.cue_horizontal = self.event_canvas.create_line(
            cue_x - 11, cue_y, cue_x + 11, cue_y, fill="#ff9f0a", width=2
        )
        self.cue_vertical = self.event_canvas.create_line(
            cue_x, cue_y - 11, cue_x, cue_y + 11, fill="#ff9f0a", width=2
        )
        self.cue_label = self.event_canvas.create_text(
            cue_x + 10,
            cue_y - 10,
            text=f"cue ({cue_head_x:g}, {cue_head_y:g})",
            fill="#ff9f0a",
            anchor="sw",
        )

        self.center_circle = self.event_canvas.create_oval(
            0, 0, 0, 0, outline="#00e5ff", width=3, state="hidden"
        )
        self.center_horizontal = self.event_canvas.create_line(
            0, 0, 0, 0, fill="#00e5ff", width=3, state="hidden"
        )
        self.center_vertical = self.event_canvas.create_line(
            0, 0, 0, 0, fill="#00e5ff", width=3, state="hidden"
        )
        self.center_label = self.event_canvas.create_text(
            0,
            0,
            text="DS centre",
            fill="#00e5ff",
            anchor="sw",
            state="hidden",
        )

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        ttk.Label(right, textvariable=self.threshold_text).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(right, textvariable=self.result_text, justify="left").grid(
            row=1, column=0, sticky="w", pady=(4, 8)
        )
        self.d_canvas = tk.Canvas(
            right,
            width=self.HISTOGRAM_WIDTH,
            height=self.HISTOGRAM_HEIGHT,
            background="#09101a",
            highlightthickness=0,
        )
        self.d_canvas.grid(row=2, column=0, sticky="ew")
        self.s_canvas = tk.Canvas(
            right,
            width=self.HISTOGRAM_WIDTH,
            height=self.HISTOGRAM_HEIGHT,
            background="#09101a",
            highlightthickness=0,
        )
        self.s_canvas.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(right, textvariable=self.runtime_text, justify="left").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )

        self._update_threshold_text()
        self.root.after(self.frame_interval_ms, self.refresh)

    def _update_threshold_text(self):
        self.threshold_text.set(
            f"Peak threshold: {self.threshold_state.get():.2f}    "
            "[ decrease    ] increase    Q/Esc quit"
        )

    def change_threshold(self, amount):
        self.threshold_state.change(amount)
        self._update_threshold_text()
        self._update_result_and_marker()

    def _drain_latest_snapshot(self):
        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.snapshot = latest

    def _event_rgb(self):
        snapshot = self.snapshot
        if snapshot is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        extra_age_us = max(
            0.0, time.monotonic() - snapshot.published_wall_time
        ) * 1_000_000.0
        age_us = (
            float(snapshot.timestamp[-1])
            - snapshot.timestamp.astype(np.float64)
            + extra_age_us
        )
        weights = np.exp(-age_us / self.fade_tau_us).astype(np.float32)
        directions = FEATURE_TO_DIRECTION[snapshot.feature]
        flat_index = (
            directions.astype(np.int32) * IMAGE_SIZE * IMAGE_SIZE
            + snapshot.y.astype(np.int32) * IMAGE_SIZE
            + snapshot.x.astype(np.int32)
        )
        heat = np.bincount(
            flat_index,
            weights=weights,
            minlength=4 * IMAGE_SIZE * IMAGE_SIZE,
        ).reshape(4, IMAGE_SIZE, IMAGE_SIZE)
        intensity = 1.0 - np.exp(-1.25 * heat)
        rgb = np.einsum("dyx,dc->yxc", intensity, DIRECTION_RGB, optimize=True)
        rgb += np.asarray((3.0, 6.0, 10.0), dtype=np.float32)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _draw_event_image(self):
        rgb = self._event_rgb()
        ppm = f"P6\n{IMAGE_SIZE} {IMAGE_SIZE}\n255\n".encode("ascii") + rgb.tobytes()
        self.base_photo.configure(data=ppm, format="PPM")
        self.scaled_photo.tk.call(
            str(self.scaled_photo),
            "copy",
            str(self.base_photo),
            "-zoom",
            self.SCALE,
            self.SCALE,
        )
        self.event_canvas.itemconfigure(self.image_item, image=self.scaled_photo)

    def _draw_histogram(self, canvas, histogram, value_offset, title, peak):
        canvas.delete("all")
        width = max(canvas.winfo_width(), self.HISTOGRAM_WIDTH)
        height = max(canvas.winfo_height(), self.HISTOGRAM_HEIGHT)
        left, right, top, bottom = 46.0, width - 12.0, 24.0, height - 30.0
        canvas.create_line(left, bottom, right, bottom, fill="#718096")
        canvas.create_line(left, top, left, bottom, fill="#718096")
        smooth = np.convolve(histogram, SMOOTH_KERNEL, mode="same")
        maximum = max(float(smooth.max()), 1.0)
        points = []
        for index, value in enumerate(smooth):
            x = left + index / 254.0 * (right - left)
            y = bottom - float(value) / maximum * (bottom - top)
            points.extend((x, y))
        canvas.create_line(*points, fill="#43b5e8", width=2)
        peak_x = left + (peak.value - value_offset) / 254.0 * (right - left)
        canvas.create_line(peak_x, top, peak_x, bottom, fill="#ffd60a", width=2)
        canvas.create_text(
            left,
            4,
            text=(
                f"{title}   peak={peak.value:.1f}   support={peak.support:.3f}   "
                f"prominence={peak.prominence:.3f}   score={peak.score:.3f}"
            ),
            fill="#e6edf3",
            anchor="nw",
        )
        canvas.create_text(left, height - 7, text=str(value_offset), fill="#aab7c4")
        canvas.create_text(
            right, height - 7, text=str(value_offset + 254), fill="#aab7c4"
        )

    def _prediction_marker_items(self):
        return (
            self.center_circle,
            self.center_horizontal,
            self.center_vertical,
            self.center_label,
        )

    def _update_result_and_marker(self):
        if self.snapshot is None:
            return
        result = self.snapshot.result
        threshold = self.threshold_state.get()
        accepted = (
            result.enough_events
            and result.inside_view
            and result.score >= threshold
        )
        state = "DETECTED; 3 s hold refreshed" if accepted else "below threshold"
        self.result_text.set(
            f"DS score={result.score:.3f}  threshold={threshold:.3f}  {state}\n"
            f"D={result.d_peak.value:.1f} (n={result.d_event_count})    "
            f"S={result.s_peak.value:.1f} (n={result.s_event_count})\n"
            f"centre=({result.cx:.1f}, {result.cy:.1f})    "
            f"window={WINDOW_EVENTS}, step={UPDATE_STEP_EVENTS}, "
            f"update={self.snapshot.update_count:,}"
        )
        if accepted:
            self.last_prediction = (result.cx, result.cy, result.score)
            self.last_prediction_wall_time = time.monotonic()
        self._update_prediction_marker()

    def _update_prediction_marker(self):
        """Keep the last threshold-passing D/S centre visible for 3 seconds."""

        marker_items = self._prediction_marker_items()
        if self.last_prediction is None:
            for item in marker_items:
                self.event_canvas.itemconfigure(item, state="hidden")
            return
        elapsed = time.monotonic() - self.last_prediction_wall_time
        if elapsed >= PREDICTION_HOLD_SEC:
            self.last_prediction = None
            for item in marker_items:
                self.event_canvas.itemconfigure(item, state="hidden")
            return

        cx, cy, score = self.last_prediction
        x = (cx + 0.5) * self.SCALE
        y = (cy + 0.5) * self.SCALE
        radius = 9.0
        self.event_canvas.coords(
            self.center_circle, x - radius, y - radius, x + radius, y + radius
        )
        self.event_canvas.coords(self.center_horizontal, x - 14, y, x + 14, y)
        self.event_canvas.coords(self.center_vertical, x, y - 14, x, y + 14)
        self.event_canvas.coords(self.center_label, x + 11, y - 11)
        remaining = max(0.0, PREDICTION_HOLD_SEC - elapsed)
        self.event_canvas.itemconfigure(
            self.center_label,
            text=f"DS ({cx:.1f}, {cy:.1f})  {score:.3f}  {remaining:.1f}s",
        )
        for item in marker_items:
            self.event_canvas.itemconfigure(item, state="normal")
            self.event_canvas.tag_raise(item)
        # The fixed cue-head marker must also stay above the refreshed image.
        for item in (
            self.cue_circle,
            self.cue_horizontal,
            self.cue_vertical,
            self.cue_label,
        ):
            self.event_canvas.tag_raise(item)

    def refresh(self):
        if self.stop_event.is_set():
            return
        self._drain_latest_snapshot()
        self._draw_event_image()
        self._update_prediction_marker()
        if (
            self.snapshot is not None
            and self.snapshot.update_count != self.last_histogram_update
        ):
            self.last_histogram_update = self.snapshot.update_count
            self._draw_histogram(
                self.d_canvas,
                self.snapshot.d_histogram,
                -127,
                "D = y - x",
                self.snapshot.result.d_peak,
            )
            self._draw_histogram(
                self.s_canvas,
                self.snapshot.s_histogram,
                0,
                "S = x + y",
                self.snapshot.result.s_peak,
            )
            self._update_result_and_marker()

        state, event_rate, update_rate, error = self.runtime_status.read()
        self.runtime_text.set(
            f"hardware: {state}    Layer-4: {event_rate:,.1f} event/s    "
            f"DS: {update_rate:,.1f} update/s"
            + (f"\n{error}" if error else "")
        )
        self.root.after(self.frame_interval_ms, self.refresh)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time 200-event / 6-event-step D/S centre demo"
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.15,
        help="minimum combined D/S peak score in 0..1 (default: 0.15)",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="maximum GUI refresh rate (default: 30)",
    )
    parser.add_argument(
        "--fade-tau-ms",
        type=float,
        default=80.0,
        help="event display fading time constant (default: 80 ms)",
    )
    parser.add_argument(
        "--cue-x",
        type=float,
        default=DEFAULT_CUE_HEAD_X,
        help="fixed cue-head x position in 128x128 coordinates (default: 68)",
    )
    parser.add_argument(
        "--cue-y",
        type=float,
        default=DEFAULT_CUE_HEAD_Y,
        help="fixed cue-head y position in 128x128 coordinates (default: 83)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 <= args.peak_threshold <= 1.0:
        raise SystemExit("--peak-threshold must be in 0..1")
    if not (0.0 <= args.cue_x < IMAGE_SIZE and 0.0 <= args.cue_y < IMAGE_SIZE):
        raise SystemExit("--cue-x and --cue-y must be in 0..127")

    snapshots = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    threshold_state = ThresholdState(args.peak_threshold)
    runtime_status = SharedRuntimeStatus()
    worker = HardwareWorker(snapshots, stop_event, runtime_status)

    root = tk.Tk()
    DSViewer(
        root,
        snapshots,
        stop_event,
        threshold_state,
        runtime_status,
        display_fps=args.display_fps,
        fade_tau_ms=args.fade_tau_ms,
        cue_head_x=args.cue_x,
        cue_head_y=args.cue_y,
    )
    worker.start()
    try:
        root.mainloop()
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
