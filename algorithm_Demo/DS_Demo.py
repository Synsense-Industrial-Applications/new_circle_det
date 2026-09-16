"""Real-time time-decayed D/S centre-vote demo for the current Layer-4 SNN.

The hardware/network configuration is imported from ``Demo.py`` so there is
only one SNN definition to maintain.  Every decoded Layer-4 event updates one
exponentially decayed D/S histogram bin and immediately recalculates the D/S
peaks.  There is no fixed-count detection window.

Only the two one-dimensional histogram peaks are used here.  Radius fitting,
circle coverage, tracking and the existing confidence pipeline are purposely
not part of this demo.

Run:
    python DS_Demo.py
    python DS_Demo.py --peak-threshold 0.14 --release-threshold 0.10

Keys:
    [ / ]       decrease / increase both hysteresis thresholds by 0.02
    Q or Esc     close the demo

Crossing the high threshold starts a detection episode; falling below the low
threshold ends it.  During the episode only the highest-confidence event frame
is retained in the worker.  It is published once when the episode ends, then
remains frozen until the next completed episode replaces it.  The Tk display is
therefore not continuously redrawn.  The fixed cue-head reference defaults to
(68, 83) and can be changed with ``--cue-x`` and ``--cue-y``.

The UI uses Tk (Python standard library) and runs in the main thread.  A second
samnagui process displays the raw Layer-4 activity through the existing samna
route from ``Demo.py``.  Hardware reading and D/S calculation run in a worker
thread.  A one-element queue keeps only the newest snapshot, so a burst of
events cannot build a rendering backlog.  The implementation contains no
Windows-only input or display API and runs on Linux when Tk, NumPy, samna and
the Speck2f runtime are installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import traceback

import numpy as np

try:
    from .layer4_layout import (
        D_HOUGH_FEATURES,
        FEATURE_TO_DIRECTION as LAYER4_FEATURE_TO_DIRECTION,
        IMAGE_SIZE,
        S_HOUGH_FEATURES,
        decode_layer4_address as _decode_layer4_address,
    )
except ImportError:  # Direct execution from algorithm_Demo/.
    from layer4_layout import (
        D_HOUGH_FEATURES,
        FEATURE_TO_DIRECTION as LAYER4_FEATURE_TO_DIRECTION,
        IMAGE_SIZE,
        S_HOUGH_FEATURES,
        decode_layer4_address as _decode_layer4_address,
    )


# This buffer is only for the Tk overlay.  At the recorded peak rate it covers
# about 120 ms, longer than the default 80 ms display fade, while keeping the
# selected best-frame snapshot inexpensive.  samnagui receives every Layer-4
# event independently of this presentation buffer.
DISPLAY_EVENT_LIMIT = 512
READ_BATCH_SIZE = 512
READ_TIMEOUT_MS = 10
DEFAULT_DECAY_TAU_MS = 400.0
DEFAULT_EVIDENCE_W0 = 20.0
DEFAULT_PEAK_THRESHOLD = 0.14
DEFAULT_RELEASE_THRESHOLD = 0.10
LAZY_SCALE_RENORMALIZE = 1e-6
TIMESTAMP_WRAP = 1 << 32

D_FAMILY = D_HOUGH_FEATURES
S_FAMILY = S_HOUGH_FEATURES
FEATURE_TO_DIRECTION = np.asarray(LAYER4_FEATURE_TO_DIRECTION, dtype=np.int8)
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
THRESHOLD_STEP = 0.02
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
    d_weight: float
    s_weight: float
    d_evidence: float
    s_evidence: float
    balance: float
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
    decay_tau_us: float
    evidence_w0: float
    detection_id: int


class ThresholdState:
    """Thread-safe, live-adjustable hysteresis thresholds."""

    def __init__(self, on_value: float, off_value: float):
        self._lock = threading.Lock()
        self._on = float(np.clip(on_value, 0.0, 1.0))
        self._off = float(np.clip(off_value, 0.0, self._on))

    def get(self) -> float:
        with self._lock:
            return self._on

    def get_pair(self):
        with self._lock:
            return self._on, self._off

    def change(self, amount: float):
        with self._lock:
            gap = self._on - self._off
            self._on = float(np.clip(self._on + amount, 0.0, 1.0))
            self._off = float(np.clip(self._on - gap, 0.0, self._on))
            return self._on, self._off


class SharedRuntimeStatus:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = "starting"
        self.layer4_rate = 0.0
        self.update_rate = 0.0
        self.detection_state = "waiting"
        self.current_confidence = 0.0
        self.detection_count = 0
        self.error = ""

    def update(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def read(self):
        with self._lock:
            return (
                self.state,
                self.layer4_rate,
                self.update_rate,
                self.detection_state,
                self.current_confidence,
                self.detection_count,
                self.error,
            )


def decode_layer4_address(x64: int, y64: int, feature: int):
    """Decode the current 64x64x16 Layer-4 address to one 128x128 point."""

    return _decode_layer4_address(x64, y64, feature)


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


def _combine_confidence(
    d_shape: float,
    s_shape: float,
    d_weight: float,
    s_weight: float,
    evidence_w0: float,
):
    """Return harmonic D/S confidence and its live evidence components.

    Shape terms are scale-invariant, so an absolute evidence term is required
    for confidence to fall during a quiet period.  The harmonic mean penalises
    a weak direction family smoothly; ``sqrt(balance)`` adds a smaller penalty
    when almost all effective events belong to only one family.
    """

    d_weight = max(float(d_weight), 0.0)
    s_weight = max(float(s_weight), 0.0)
    d_evidence = 1.0 - math.exp(-d_weight / evidence_w0)
    s_evidence = 1.0 - math.exp(-s_weight / evidence_w0)
    d_family = float(d_shape) * float(d_evidence)
    s_family = float(s_shape) * float(s_evidence)
    balance = 2.0 * min(d_weight, s_weight) / max(d_weight + s_weight, 1e-12)
    harmonic = 2.0 * d_family * s_family / max(d_family + s_family, 1e-12)
    score = math.sqrt(balance) * harmonic
    return (
        float(np.clip(score, 0.0, 1.0)),
        float(d_evidence),
        float(s_evidence),
        float(balance),
    )


def confidence_at_age(
    result: DSResult,
    age_us: float,
    tau_us: float,
    evidence_w0: float,
):
    """Evaluate current confidence without waiting for another input event."""

    decay = math.exp(-max(float(age_us), 0.0) / max(float(tau_us), 1.0))
    d_weight = result.d_weight * decay
    s_weight = result.s_weight * decay
    score, d_evidence, s_evidence, balance = _combine_confidence(
        result.d_peak.score,
        result.s_peak.score,
        d_weight,
        s_weight,
        evidence_w0,
    )
    return score, d_weight, s_weight, d_evidence, s_evidence, balance


class DSDecayAccumulator:
    """Event-triggered exponentially decayed D/S accumulator.

    ``_global_scale`` applies the same decay to every histogram bin.  New
    events are inserted divided by that scale, so the per-event update touches
    one scalar and one bin instead of multiplying two 255-bin arrays.  The
    display history is a separate bounded buffer and is not used for detection.
    """

    def __init__(
        self,
        decay_tau_ms=DEFAULT_DECAY_TAU_MS,
        evidence_w0=DEFAULT_EVIDENCE_W0,
    ):
        self.decay_tau_us = max(float(decay_tau_ms) * 1000.0, 1.0)
        self.evidence_w0 = max(float(evidence_w0), 1e-6)
        self._d_histogram = np.zeros(255, dtype=np.float64)
        self._s_histogram = np.zeros(255, dtype=np.float64)
        self._global_scale = 1.0
        self._last_raw_timestamp = None
        self._logical_timestamp_us = 0
        self._display_x = np.empty(DISPLAY_EVENT_LIMIT, dtype=np.int16)
        self._display_y = np.empty(DISPLAY_EVENT_LIMIT, dtype=np.int16)
        self._display_feature = np.empty(DISPLAY_EVENT_LIMIT, dtype=np.int8)
        self._display_timestamp = np.empty(DISPLAY_EVENT_LIMIT, dtype=np.int64)
        self._display_write = 0
        self._display_count = 0
        self.source_event_count = 0
        self.update_count = 0

    @staticmethod
    def _timestamp_delta(previous: int | None, current: int) -> int:
        if previous is None:
            return 0
        delta = int(current) - int(previous)
        if delta >= 0:
            return delta
        # samna timestamps are commonly uint32 microseconds.  Treat a high-to-
        # low transition as wraparound; otherwise tolerate a reordered event.
        if int(previous) > TIMESTAMP_WRAP // 2 and int(current) < TIMESTAMP_WRAP // 2:
            return TIMESTAMP_WRAP - int(previous) + int(current)
        return 0

    def _advance_time(self, raw_timestamp: int):
        delta_us = self._timestamp_delta(self._last_raw_timestamp, raw_timestamp)
        self._last_raw_timestamp = int(raw_timestamp)
        self._logical_timestamp_us += delta_us
        self._global_scale *= math.exp(-delta_us / self.decay_tau_us)
        if self._global_scale < LAZY_SCALE_RENORMALIZE:
            self._d_histogram *= self._global_scale
            self._s_histogram *= self._global_scale
            self._global_scale = 1.0

    def push_raw(self, x64, y64, feature, timestamp):
        x, y = decode_layer4_address(x64, y64, feature)
        feature = int(feature)
        self._advance_time(int(timestamp))
        reciprocal_scale = 1.0 / self._global_scale
        if feature in D_FAMILY:
            self._d_histogram[y - x + 127] += reciprocal_scale
        else:
            self._s_histogram[x + y] += reciprocal_scale
        display_index = self._display_write
        self._display_x[display_index] = x
        self._display_y[display_index] = y
        self._display_feature[display_index] = feature
        self._display_timestamp[display_index] = self._logical_timestamp_us
        self._display_write = (display_index + 1) % DISPLAY_EVENT_LIMIT
        self._display_count = min(self._display_count + 1, DISPLAY_EVENT_LIMIT)
        self.source_event_count += 1
        self.update_count += 1
        return self._calculate_result()

    def _display_snapshot(self):
        if self._display_count < DISPLAY_EVENT_LIMIT:
            stop = self._display_count
            return (
                self._display_x[:stop].copy(),
                self._display_y[:stop].copy(),
                self._display_feature[:stop].copy(),
                self._display_timestamp[:stop].copy(),
            )
        start = self._display_write
        if start == 0:
            return (
                self._display_x.copy(),
                self._display_y.copy(),
                self._display_feature.copy(),
                self._display_timestamp.copy(),
            )
        return tuple(
            np.concatenate((values[start:], values[:start]))
            for values in (
                self._display_x,
                self._display_y,
                self._display_feature,
                self._display_timestamp,
            )
        )

    def _calculate_result(self):
        # Peak shape and position are invariant under the common global scale,
        # so no 255-bin array copy is needed on ordinary event updates.
        d_peak = _peak_from_histogram(self._d_histogram, -127)
        s_peak = _peak_from_histogram(self._s_histogram, 0)
        if np.isfinite(d_peak.value + s_peak.value):
            cx = (s_peak.value - d_peak.value) / 2.0
            cy = (s_peak.value + d_peak.value) / 2.0
        else:
            cx = cy = np.nan

        d_weight = float(self._d_histogram.sum()) * self._global_scale
        s_weight = float(self._s_histogram.sum()) * self._global_scale
        score, d_evidence, s_evidence, balance = _combine_confidence(
            d_peak.score,
            s_peak.score,
            d_weight,
            s_weight,
            self.evidence_w0,
        )
        return DSResult(
            cx=float(cx),
            cy=float(cy),
            d_peak=d_peak,
            s_peak=s_peak,
            score=score,
            d_weight=d_weight,
            s_weight=s_weight,
            d_evidence=d_evidence,
            s_evidence=s_evidence,
            balance=balance,
            inside_view=bool(0.0 <= cx < IMAGE_SIZE and 0.0 <= cy < IMAGE_SIZE),
            enough_events=bool(d_weight > 1e-6 and s_weight > 1e-6),
        )

    def make_snapshot(self, result: DSResult, detection_id: int):
        """Copy presentation data only when a new best frame is selected."""

        d_histogram = (self._d_histogram * self._global_scale).astype(np.float32)
        s_histogram = (self._s_histogram * self._global_scale).astype(np.float32)
        display_x, display_y, display_feature, display_timestamp = (
            self._display_snapshot()
        )
        return DSSnapshot(
            x=display_x,
            y=display_y,
            feature=display_feature,
            timestamp=display_timestamp,
            d_histogram=d_histogram,
            s_histogram=s_histogram,
            result=result,
            source_event_count=self.source_event_count,
            update_count=self.update_count,
            published_wall_time=time.monotonic(),
            decay_tau_us=self.decay_tau_us,
            evidence_w0=self.evidence_w0,
            detection_id=int(detection_id),
        )


class HysteresisPeakHold:
    """Select and hold the highest-confidence frame in each detection episode."""

    def __init__(self):
        self.active = False
        self.detection_count = 0
        self.best_score = -np.inf

    @staticmethod
    def _valid_score(result: DSResult) -> float:
        if not result.enough_events or not result.inside_view:
            return -np.inf
        return float(result.score)

    def observe(self, result: DSResult, on_threshold: float, off_threshold: float):
        """Return ``(new_best, episode_ended)`` for this event."""

        score = self._valid_score(result)
        episode_ended = False
        if self.active and score < off_threshold:
            self.active = False
            episode_ended = True

        if not self.active:
            if score < on_threshold:
                return False, episode_ended
            self.active = True
            self.detection_count += 1
            self.best_score = score
            return True, episode_ended

        if score > self.best_score:
            self.best_score = score
            return True, episode_ended
        return False, episode_ended

    def poll(self, current_score: float, off_threshold: float):
        """End an active episode during a no-event interval."""

        if self.active and current_score < off_threshold:
            self.active = False
            return True
        return False

    @property
    def state_label(self):
        if self.active:
            return f"ACTIVE #{self.detection_count}: selecting peak frame"
        if self.detection_count:
            return f"HOLD #{self.detection_count}: waiting for next centre"
        return "WAITING: confidence below C_on"


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

    def __init__(
        self,
        output_queue,
        stop_event,
        runtime_status,
        threshold_state,
        decay_tau_ms,
        evidence_w0,
        show_samna_layer4=True,
    ):
        super().__init__(name="ds-hardware", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.runtime_status = runtime_status
        self.threshold_state = threshold_state
        self.decay_tau_ms = float(decay_tau_ms)
        self.evidence_w0 = float(evidence_w0)
        self.show_samna_layer4 = bool(show_samna_layer4)

    def run(self):
        input_graph = None
        device_input_route = None
        viz_graph = None
        viz_gui = None
        try:
            # Delay hardware imports so --help and offline algorithm tests work
            # on development machines without the Speck2f runtime installed.
            import samna
            from Demo import (
                config,
                configure_cnn_pipeline,
                layer_4,
                open_speck2f_dev_kit,
                visualize_layer,
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

            if self.show_samna_layer4:
                self.runtime_status.update(state="opening samna Layer-4 viewer")
                viz_graph, viz_gui = visualize_layer(dev_kit, layer_4)

            event_buffer.get_events()
            detector = DSDecayAccumulator(
                decay_tau_ms=self.decay_tau_ms,
                evidence_w0=self.evidence_w0,
            )
            peak_hold = HysteresisPeakHold()
            last_result = None
            last_event_wall_time = None
            pending_best_snapshot = None
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
                        result = detector.push_raw(
                            getattr(event, "x"),
                            getattr(event, "y"),
                            getattr(event, "feature"),
                            getattr(event, "timestamp"),
                        )
                    except (AttributeError, TypeError, ValueError):
                        continue
                    last_result = result
                    last_event_wall_time = time.monotonic()
                    on_threshold, off_threshold = self.threshold_state.get_pair()
                    new_best, episode_ended = peak_hold.observe(
                        result, on_threshold, off_threshold
                    )
                    if new_best:
                        pending_best_snapshot = detector.make_snapshot(
                            result, peak_hold.detection_count
                        )
                    if episode_ended and pending_best_snapshot is not None:
                        publish_latest(
                            self.output_queue, pending_best_snapshot
                        )
                        pending_best_snapshot = None

                now = time.monotonic()
                current_confidence = 0.0
                if last_result is not None and last_event_wall_time is not None:
                    age_us = max(0.0, now - last_event_wall_time) * 1_000_000.0
                    current_confidence = confidence_at_age(
                        last_result,
                        age_us,
                        detector.decay_tau_us,
                        detector.evidence_w0,
                    )[0]
                    if not last_result.enough_events or not last_result.inside_view:
                        current_confidence = 0.0
                _on_threshold, off_threshold = self.threshold_state.get_pair()
                if peak_hold.poll(current_confidence, off_threshold):
                    if pending_best_snapshot is not None:
                        publish_latest(
                            self.output_queue, pending_best_snapshot
                        )
                        pending_best_snapshot = None
                self.runtime_status.update(
                    detection_state=peak_hold.state_label,
                    current_confidence=current_confidence,
                    detection_count=peak_hold.detection_count,
                )
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
            if viz_graph is not None:
                try:
                    viz_graph.stop()
                except Exception:
                    pass
            if viz_gui is not None:
                try:
                    viz_gui.terminate()
                    viz_gui.join(timeout=2.0)
                except Exception:
                    pass
            # Intentional lifetime anchor for the device route.
            _ = device_input_route
            if not self.runtime_status.read()[6]:
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
        self.base_photo = tk.PhotoImage(width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.scaled_photo = tk.PhotoImage(width=self.EVENT_SIDE, height=self.EVENT_SIDE)
        self.image_item = None

        self.threshold_text = tk.StringVar()
        self.result_text = tk.StringVar(
            value="Waiting for confidence to cross C_on..."
        )
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
        ttk.Label(
            left, text="Held best-confidence detection frame (128 x 128)"
        ).pack(anchor="w")
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
        ttk.Label(
            right,
            justify="left",
            text=(
                "C = sqrt(B) * 2*F_D*F_S/(F_D+F_S)\n"
                "F_f = shape_f * (1-exp(-W_f/W0)); C_on/C_off hysteresis"
            ),
        ).grid(row=2, column=0, sticky="w", pady=(0, 8))
        self.d_canvas = tk.Canvas(
            right,
            width=self.HISTOGRAM_WIDTH,
            height=self.HISTOGRAM_HEIGHT,
            background="#09101a",
            highlightthickness=0,
        )
        self.d_canvas.grid(row=3, column=0, sticky="ew")
        self.s_canvas = tk.Canvas(
            right,
            width=self.HISTOGRAM_WIDTH,
            height=self.HISTOGRAM_HEIGHT,
            background="#09101a",
            highlightthickness=0,
        )
        self.s_canvas.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(right, textvariable=self.runtime_text, justify="left").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )

        self._update_threshold_text()
        self.root.after(self.frame_interval_ms, self.refresh)

    def _update_threshold_text(self):
        on_threshold, off_threshold = self.threshold_state.get_pair()
        self.threshold_text.set(
            f"Hysteresis: C_on={on_threshold:.2f}  C_off={off_threshold:.2f}    "
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
            return True
        return False

    def _event_rgb(self):
        snapshot = self.snapshot
        if snapshot is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        age_us = (
            float(snapshot.timestamp[-1])
            - snapshot.timestamp.astype(np.float64)
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

    def _draw_histogram(
        self, canvas, histogram, value_offset, title, peak, weight, evidence
    ):
        canvas.delete("all")
        width = max(canvas.winfo_width(), self.HISTOGRAM_WIDTH)
        height = max(canvas.winfo_height(), self.HISTOGRAM_HEIGHT)
        left, right, top, bottom = 46.0, width - 12.0, 42.0, height - 30.0
        canvas.create_line(left, bottom, right, bottom, fill="#718096")
        canvas.create_line(left, top, left, bottom, fill="#718096")
        smooth = np.convolve(histogram, SMOOTH_KERNEL, mode="same")
        # A floor keeps absolute fading visible instead of renormalising a tiny
        # stale histogram back to full height.
        maximum = max(float(smooth.max()), 1.0)
        points = []
        for index, value in enumerate(smooth):
            x = left + index / 254.0 * (right - left)
            y = bottom - float(value) / maximum * (bottom - top)
            points.extend((x, y))
        brightness = int(np.clip(90 + 165 * evidence, 0, 255))
        line_colour = (
            f"#{int(0.27 * brightness):02x}"
            f"{int(0.71 * brightness):02x}{brightness:02x}"
        )
        canvas.create_line(*points, fill=line_colour, width=2)
        if np.isfinite(peak.value):
            peak_x = left + (peak.value - value_offset) / 254.0 * (right - left)
            canvas.create_line(
                peak_x, top, peak_x, bottom, fill="#ffd60a", width=2
            )
        canvas.create_text(
            left,
            4,
            text=(
                f"{title}  peak={peak.value:.1f}  W={weight:.2f}  E={evidence:.3f}\n"
                f"shape={peak.score:.3f}  support={peak.support:.3f}  "
                f"prominence={peak.prominence:.3f}"
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

    def _update_result_and_marker(self, redraw_marker=False):
        if self.snapshot is None:
            return
        result = self.snapshot.result
        (
            _hardware_state,
            _event_rate,
            _update_rate,
            detection_state,
            current_confidence,
            _detection_count,
            _error,
        ) = self.runtime_status.read()
        self.result_text.set(
            f"held detection #{self.snapshot.detection_id}: "
            f"best confidence={result.score:.3f}\n"
            f"current confidence={current_confidence:.3f}  {detection_state}\n"
            f"D={result.d_peak.value:.1f} "
            f"(W={result.d_weight:.2f}, E={result.d_evidence:.3f})    "
            f"S={result.s_peak.value:.1f} "
            f"(W={result.s_weight:.2f}, E={result.s_evidence:.3f})\n"
            f"held centre=({result.cx:.1f}, {result.cy:.1f})    "
            f"balance={result.balance:.3f}\n"
            f"tau={self.snapshot.decay_tau_us / 1000.0:.0f} ms, "
            f"W0={self.snapshot.evidence_w0:g}, every event, "
            f"event={self.snapshot.source_event_count:,}"
        )
        if not redraw_marker:
            return
        marker_items = self._prediction_marker_items()
        x = (result.cx + 0.5) * self.SCALE
        y = (result.cy + 0.5) * self.SCALE
        radius = 9.0
        self.event_canvas.coords(
            self.center_circle, x - radius, y - radius, x + radius, y + radius
        )
        self.event_canvas.coords(self.center_horizontal, x - 14, y, x + 14, y)
        self.event_canvas.coords(self.center_vertical, x, y - 14, x, y + 14)
        self.event_canvas.coords(self.center_label, x + 11, y - 11)
        self.event_canvas.itemconfigure(
            self.center_label,
            text=(
                f"DS ({result.cx:.1f}, {result.cy:.1f})  "
                f"best C={result.score:.3f}"
            ),
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
        has_new_best_frame = self._drain_latest_snapshot()
        if self.snapshot is not None:
            if has_new_best_frame:
                self._draw_event_image()
                result = self.snapshot.result
                self._draw_histogram(
                    self.d_canvas,
                    self.snapshot.d_histogram,
                    -127,
                    "D = y - x",
                    result.d_peak,
                    result.d_weight,
                    result.d_evidence,
                )
                self._draw_histogram(
                    self.s_canvas,
                    self.snapshot.s_histogram,
                    0,
                    "S = x + y",
                    result.s_peak,
                    result.s_weight,
                    result.s_evidence,
                )
            self._update_result_and_marker(redraw_marker=has_new_best_frame)

        (
            state,
            event_rate,
            update_rate,
            detection_state,
            current_confidence,
            detection_count,
            error,
        ) = self.runtime_status.read()
        self.runtime_text.set(
            f"hardware: {state}    Layer-4: {event_rate:,.1f} event/s    "
            f"DS: {update_rate:,.1f} update/s\n"
            f"{detection_state}    current C={current_confidence:.3f}    "
            f"detections={detection_count}"
            + (f"\n{error}" if error else "")
        )
        self.root.after(self.frame_interval_ms, self.refresh)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time event-triggered, time-decayed D/S centre demo"
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD,
        help="C_on: confidence that starts a detection episode (default: 0.14)",
    )
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=DEFAULT_RELEASE_THRESHOLD,
        help="confidence that ends an active detection (default: 0.10)",
    )
    parser.add_argument(
        "--decay-tau-ms",
        type=float,
        default=DEFAULT_DECAY_TAU_MS,
        help="D/S event time-decay constant in milliseconds (default: 400)",
    )
    parser.add_argument(
        "--evidence-w0",
        type=float,
        default=DEFAULT_EVIDENCE_W0,
        help="effective events per family for 63%% evidence (default: 20)",
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
    parser.add_argument(
        "--no-samna-layer4",
        action="store_true",
        help="do not open the separate samnagui Layer-4 activity window",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 <= args.peak_threshold <= 1.0:
        raise SystemExit("--peak-threshold must be in 0..1")
    if not 0.0 <= args.release_threshold < args.peak_threshold:
        raise SystemExit(
            "--release-threshold must be >= 0 and below --peak-threshold"
        )
    if args.decay_tau_ms <= 0.0:
        raise SystemExit("--decay-tau-ms must be positive")
    if args.evidence_w0 <= 0.0:
        raise SystemExit("--evidence-w0 must be positive")
    if not (0.0 <= args.cue_x < IMAGE_SIZE and 0.0 <= args.cue_y < IMAGE_SIZE):
        raise SystemExit("--cue-x and --cue-y must be in 0..127")

    snapshots = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    threshold_state = ThresholdState(
        args.peak_threshold, args.release_threshold
    )
    runtime_status = SharedRuntimeStatus()
    worker = HardwareWorker(
        snapshots,
        stop_event,
        runtime_status,
        threshold_state,
        decay_tau_ms=args.decay_tau_ms,
        evidence_w0=args.evidence_w0,
        show_samna_layer4=not args.no_samna_layer4,
    )

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
        worker.join(timeout=4.0)


if __name__ == "__main__":
    main()
