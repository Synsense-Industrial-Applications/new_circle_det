"""Hit timing and non-blocking replay for the hardware circle demo.

The realtime detector itself stays responsible for circle geometry.  This
module adds three independent pieces around it:

1. :class:`RadiusPeakHitDetector` finds a causal peak in the accepted radius
   sequence (or a rising sequence that is immediately lost).
2. :class:`SlidingEventWindow` retains recent decoded events and freezes a
   clip around every trigger.
3. :class:`HitReplayViewer` plays clips in a separate process so rendering
   never stalls hardware event collection.

Timestamps are expressed in the same unit as ``FlowEvent.t``.  The Speck2f
demo and repository CSV files use microseconds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import multiprocessing
import queue
import time
from typing import Deque, Optional, Sequence

import numpy as np

from circle_detection import FlowEvent


@dataclass(frozen=True)
class HitReplayConfig:
    """All timing, buffering and display parameters in one place."""

    enabled: bool = True
    cue_x_px: float = 68.0
    cue_y_px: float = 83.0
    timing_required_filter_rules: tuple[str, ...] = (
        "radius",
        "center_x",
        "center_y",
    )
    min_timing_confidence: float = 0.15

    # Radius peak detector.  AdaptiveCircleDetector already smooths geometry;
    # this EMA removes the small residual jitter used for timing decisions.
    radius_ema_alpha: float = 0.45
    rise_window_samples: int = 24
    fall_confirm_samples: int = 3
    max_peak_age_samples: int = 30
    min_radius_rise_px: float = 0.80
    min_radius_fall_px: float = 0.55
    min_rising_fraction: float = 0.60
    min_falling_fraction: float = 0.67
    min_peak_radius_px: float = 35.0

    # A strike can make the tracked ball disappear before a falling radius is
    # observed.  This second path requires a stronger rise and a large radius.
    detect_rise_then_lost: bool = True
    lost_confirm_updates: int = 3
    lost_min_radius_px: float = 38.0
    lost_min_radius_rise_px: float = 1.00
    cooldown_us: float = 800_000.0

    # Sliding event window and replay timing.
    pre_hit_us: float = 150_000.0
    post_hit_us: float = 200_000.0
    buffer_duration_us: float = 700_000.0
    max_buffer_events: int = 60_000
    replay_speed: float = 0.20
    replay_fps: float = 30.0
    event_trail_us: float = 35_000.0
    final_hold_sec: float = 1.5
    replay_queue_size: int = 2

    def __post_init__(self) -> None:
        finite_values = (
            self.cue_x_px,
            self.cue_y_px,
            self.min_timing_confidence,
            self.radius_ema_alpha,
            self.min_radius_rise_px,
            self.min_radius_fall_px,
            self.min_rising_fraction,
            self.min_falling_fraction,
            self.min_peak_radius_px,
            self.lost_min_radius_px,
            self.lost_min_radius_rise_px,
            self.cooldown_us,
            self.pre_hit_us,
            self.post_hit_us,
            self.buffer_duration_us,
            self.replay_speed,
            self.replay_fps,
            self.event_trail_us,
            self.final_hold_sec,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("hit replay parameters must be finite")
        if not 0.0 < self.radius_ema_alpha <= 1.0:
            raise ValueError("radius_ema_alpha must be in (0, 1]")
        if not 0.0 <= self.min_timing_confidence <= 1.0:
            raise ValueError("min_timing_confidence must be in [0, 1]")
        if (
            not self.timing_required_filter_rules
            or len(set(self.timing_required_filter_rules))
            != len(self.timing_required_filter_rules)
        ):
            raise ValueError("timing filter rule names must be nonempty and unique")
        if self.rise_window_samples < 2 or self.fall_confirm_samples < 1:
            raise ValueError("radius peak windows are too short")
        if self.max_peak_age_samples < self.fall_confirm_samples:
            raise ValueError("max_peak_age_samples must cover fall confirmation")
        if not 0.0 <= self.min_rising_fraction <= 1.0:
            raise ValueError("min_rising_fraction must be in [0, 1]")
        if not 0.0 <= self.min_falling_fraction <= 1.0:
            raise ValueError("min_falling_fraction must be in [0, 1]")
        if self.lost_confirm_updates < 1:
            raise ValueError("lost_confirm_updates must be positive")
        if min(
            self.min_radius_rise_px,
            self.min_radius_fall_px,
            self.min_peak_radius_px,
            self.lost_min_radius_px,
            self.lost_min_radius_rise_px,
            self.cooldown_us,
            self.pre_hit_us,
            self.post_hit_us,
            self.event_trail_us,
            self.final_hold_sec,
        ) < 0.0:
            raise ValueError("hit replay thresholds and durations must be nonnegative")
        if self.buffer_duration_us < self.pre_hit_us + self.post_hit_us:
            raise ValueError("buffer_duration_us must cover pre_hit_us + post_hit_us")
        if self.max_buffer_events < 1 or self.replay_queue_size < 1:
            raise ValueError("event and replay queue sizes must be positive")
        if self.replay_speed <= 0.0 or self.replay_fps <= 0.0:
            raise ValueError("replay_speed and replay_fps must be positive")


@dataclass(frozen=True)
class RadiusSample:
    update_number: int
    source_event_count: int
    timestamp: float
    cx: float
    cy: float
    radius: float
    smoothed_radius: float
    confidence: float


@dataclass(frozen=True)
class HitTrigger:
    hit_number: int
    reason: str
    detected_timestamp: float
    peak: RadiusSample
    radius_rise_px: float
    radius_fall_px: float
    cue_x_px: float
    cue_y_px: float

    @property
    def cue_offset_x_px(self) -> float:
        return self.cue_x_px - self.peak.cx

    @property
    def cue_offset_y_px(self) -> float:
        return self.cue_y_px - self.peak.cy

    @property
    def cue_offset_x_radius(self) -> float:
        return self.cue_offset_x_px / max(self.peak.radius, 1e-9)

    @property
    def cue_offset_y_radius(self) -> float:
        return self.cue_offset_y_px / max(self.peak.radius, 1e-9)


@dataclass(frozen=True)
class HitReplayClip:
    trigger: HitTrigger
    events: tuple[FlowEvent, ...]
    requested_start_timestamp: float
    requested_end_timestamp: float
    complete: bool

    @property
    def duration_us(self) -> float:
        return max(0.0, self.requested_end_timestamp - self.requested_start_timestamp)


class RadiusPeakHitDetector:
    """Causal hit detector operating on accepted circle updates."""

    def __init__(self, config: HitReplayConfig = HitReplayConfig()) -> None:
        self.config = config
        history_size = config.rise_window_samples + config.max_peak_age_samples + 1
        self._history: Deque[RadiusSample] = deque(maxlen=history_size)
        self._ema_radius: Optional[float] = None
        self._missing_updates = 0
        self._last_hit_timestamp = -math.inf
        self.hit_count = 0

    @property
    def history(self) -> tuple[RadiusSample, ...]:
        return tuple(self._history)

    @property
    def missing_updates(self) -> int:
        return self._missing_updates

    def reset_track(self) -> None:
        self._history.clear()
        self._ema_radius = None
        self._missing_updates = 0

    def _outside_cooldown(self, timestamp: float) -> bool:
        if timestamp < self._last_hit_timestamp:
            # A device timestamp reset starts a fresh timeline.
            self._last_hit_timestamp = -math.inf
        return timestamp - self._last_hit_timestamp >= self.config.cooldown_us

    @staticmethod
    def _step_fraction(values: Sequence[float], *, rising: bool) -> float:
        differences = np.diff(np.asarray(values, dtype=float))
        if not len(differences):
            return 0.0
        matches = differences > 0.0 if rising else differences < 0.0
        return float(np.count_nonzero(matches) / len(differences))

    def _make_trigger(
        self,
        reason: str,
        detected_timestamp: float,
        peak: RadiusSample,
        rise: float,
        fall: float,
    ) -> Optional[HitTrigger]:
        if not self._outside_cooldown(peak.timestamp):
            # Discard the already-evaluated shape; otherwise its old maximum
            # would mask a later peak after the cooldown expires.
            self.reset_track()
            return None
        self.hit_count += 1
        self._last_hit_timestamp = peak.timestamp
        trigger = HitTrigger(
            hit_number=self.hit_count,
            reason=reason,
            detected_timestamp=float(detected_timestamp),
            peak=peak,
            radius_rise_px=float(rise),
            radius_fall_px=float(fall),
            cue_x_px=self.config.cue_x_px,
            cue_y_px=self.config.cue_y_px,
        )
        self.reset_track()
        return trigger

    def _detect_confirmed_peak(self) -> Optional[HitTrigger]:
        config = self.config
        required = config.rise_window_samples + config.fall_confirm_samples + 1
        if len(self._history) < required:
            return None
        samples = list(self._history)
        all_radii = [sample.smoothed_radius for sample in samples]
        peak_index = int(np.argmax(all_radii))
        samples_after_peak = len(samples) - peak_index - 1
        if (
            peak_index < config.rise_window_samples
            or samples_after_peak < config.fall_confirm_samples
            or samples_after_peak > config.max_peak_age_samples
        ):
            return None
        peak = samples[peak_index]
        left = samples[peak_index - config.rise_window_samples : peak_index + 1]
        right = samples[peak_index:]
        left_radius = [sample.smoothed_radius for sample in left]
        right_radius = [sample.smoothed_radius for sample in right]
        peak_radius = peak.smoothed_radius
        rise = peak_radius - min(left_radius[:-1])
        fall = peak_radius - min(right_radius[1:])
        rising_fraction = self._step_fraction(left_radius, rising=True)
        lowest_index = int(np.argmin(right_radius[1:])) + 1
        falling_fraction = self._step_fraction(
            right_radius[: lowest_index + 1], rising=False
        )

        if not (
            peak_radius >= config.min_peak_radius_px
            and rise >= config.min_radius_rise_px
            and fall >= config.min_radius_fall_px
            and rising_fraction >= config.min_rising_fraction
            and falling_fraction >= config.min_falling_fraction
        ):
            return None
        return self._make_trigger(
            "radius_peak",
            samples[-1].timestamp,
            peak,
            rise,
            fall,
        )

    def observe_circle(
        self,
        *,
        update_number: int,
        source_event_count: int,
        timestamp: float,
        cx: float,
        cy: float,
        radius: float,
        confidence: float,
    ) -> Optional[HitTrigger]:
        values = (timestamp, cx, cy, radius, confidence)
        if not all(math.isfinite(float(value)) for value in values):
            return self.observe_missing(timestamp=timestamp)
        alpha = self.config.radius_ema_alpha
        smoothed = (
            float(radius)
            if self._ema_radius is None
            else (1.0 - alpha) * self._ema_radius + alpha * float(radius)
        )
        self._ema_radius = smoothed
        self._missing_updates = 0
        self._history.append(
            RadiusSample(
                update_number=int(update_number),
                source_event_count=int(source_event_count),
                timestamp=float(timestamp),
                cx=float(cx),
                cy=float(cy),
                radius=float(radius),
                smoothed_radius=smoothed,
                confidence=float(confidence),
            )
        )
        return self._detect_confirmed_peak()

    def observe_missing(self, *, timestamp: float) -> Optional[HitTrigger]:
        """Record one detector update without an accepted circle."""

        self._missing_updates += 1
        config = self.config
        if self._missing_updates < config.lost_confirm_updates:
            return None

        trigger: Optional[HitTrigger] = None
        samples = list(self._history)
        required = config.rise_window_samples + 1
        if config.detect_rise_then_lost and len(samples) >= required:
            recent = samples[-required:]
            radii = [sample.smoothed_radius for sample in recent]
            peak = recent[-1]
            rise = peak.smoothed_radius - min(radii[:-1])
            rising_fraction = self._step_fraction(radii, rising=True)
            if (
                peak.smoothed_radius >= config.lost_min_radius_px
                and peak.smoothed_radius >= max(radii) - 1e-9
                and rise >= config.lost_min_radius_rise_px
                and rising_fraction >= config.min_rising_fraction
            ):
                trigger = self._make_trigger(
                    "rising_then_lost",
                    float(timestamp),
                    peak,
                    rise,
                    0.0,
                )
        if trigger is None:
            self.reset_track()
        return trigger


class SlidingEventWindow:
    """Timestamp-pruned event buffer that freezes clips after a hit."""

    def __init__(self, config: HitReplayConfig = HitReplayConfig()) -> None:
        self.config = config
        self._events: Deque[FlowEvent] = deque(maxlen=config.max_buffer_events)
        self._pending: Deque[HitTrigger] = deque()
        self._ready: Deque[HitReplayClip] = deque()
        self._latest_timestamp = -math.inf

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def push(self, event: FlowEvent) -> None:
        self._events.append(event)
        self._latest_timestamp = max(self._latest_timestamp, float(event.t))
        oldest = self._latest_timestamp - self.config.buffer_duration_us
        while self._events and self._events[0].t < oldest:
            self._events.popleft()
        self._finalize_ready(complete=True)

    def arm(self, trigger: HitTrigger) -> None:
        self._pending.append(trigger)
        self._finalize_ready(complete=True)

    def _build_clip(self, trigger: HitTrigger, *, complete: bool) -> HitReplayClip:
        start = trigger.peak.timestamp - self.config.pre_hit_us
        end = trigger.peak.timestamp + self.config.post_hit_us
        events = tuple(event for event in self._events if start <= event.t <= end)
        return HitReplayClip(trigger, events, start, end, complete)

    def _finalize_ready(self, *, complete: bool) -> None:
        while self._pending:
            trigger = self._pending[0]
            end = trigger.peak.timestamp + self.config.post_hit_us
            if complete and self._latest_timestamp < end:
                break
            self._pending.popleft()
            self._ready.append(self._build_clip(trigger, complete=complete))

    def pop_ready(self) -> tuple[HitReplayClip, ...]:
        clips = tuple(self._ready)
        self._ready.clear()
        return clips

    def flush(self) -> tuple[HitReplayClip, ...]:
        self._finalize_ready(complete=False)
        return self.pop_ready()


def format_hit_trigger(trigger: HitTrigger) -> str:
    """Stable terminal summary for one detected hit."""

    peak = trigger.peak
    return (
        f"[HIT DETECTED] hit={trigger.hit_number} reason={trigger.reason} "
        f"event={peak.source_event_count} hit_t_us={int(peak.timestamp)} "
        f"detected_t_us={int(trigger.detected_timestamp)} "
        f"ball=({peak.cx:.3f},{peak.cy:.3f}) r={peak.radius:.3f} "
        f"cue=({trigger.cue_x_px:.3f},{trigger.cue_y_px:.3f}) "
        f"offset_px=({trigger.cue_offset_x_px:+.3f},"
        f"{trigger.cue_offset_y_px:+.3f}) "
        f"offset_r=({trigger.cue_offset_x_radius:+.3f},"
        f"{trigger.cue_offset_y_radius:+.3f}) "
        f"rise={trigger.radius_rise_px:.3f} "
        f"fall={trigger.radius_fall_px:.3f} conf={peak.confidence:.4f}"
    )


def _draw_replay_frame(axes, clip: HitReplayClip, timestamp: float) -> None:
    from matplotlib.patches import Circle

    config_cue = (clip.trigger.cue_x_px, clip.trigger.cue_y_px)
    peak = clip.trigger.peak
    events = clip.events
    if events:
        x = np.fromiter((event.x for event in events), dtype=float)
        y = np.fromiter((event.y for event in events), dtype=float)
        code = np.fromiter((event.c for event in events), dtype=int)
        axes.scatter(
            x,
            y,
            c=code,
            cmap="tab10",
            vmin=0,
            vmax=9,
            s=8,
            alpha=0.72,
            linewidths=0,
        )
    axes.add_patch(
        Circle(
            (peak.cx, peak.cy),
            peak.radius,
            fill=False,
            color="#31d158",
            linewidth=2.2,
            label="detected ball",
        )
    )
    axes.plot(peak.cx, peak.cy, "+", color="#31d158", markersize=12, mew=2)
    axes.plot(
        config_cue[0],
        config_cue[1],
        "x",
        color="#ff453a",
        markersize=12,
        mew=2.5,
        label="cue (68, 83)",
    )
    axes.plot(
        [peak.cx, config_cue[0]],
        [peak.cy, config_cue[1]],
        linestyle="--",
        color="#ffd60a",
        linewidth=1.2,
    )
    relative_ms = (timestamp - peak.timestamp) / 1_000.0
    phase = "HIT" if timestamp >= peak.timestamp else "pre-hit"
    axes.set_title(
        f"Hit #{clip.trigger.hit_number}  {phase}  {relative_ms:+.1f} ms\n"
        f"cue offset / radius = "
        f"({clip.trigger.cue_offset_x_radius:+.2f}, "
        f"{clip.trigger.cue_offset_y_radius:+.2f})"
    )
    axes.set_xlim(0, 127)
    axes.set_ylim(127, 0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x (px)")
    axes.set_ylabel("y (px)")
    axes.grid(alpha=0.15)
    axes.legend(loc="upper right", fontsize=8)


def _play_clip(figure, axes, clip: HitReplayClip, config: HitReplayConfig) -> None:
    import matplotlib.pyplot as plt

    if not clip.events:
        return
    start = clip.requested_start_timestamp
    end = min(clip.requested_end_timestamp, max(event.t for event in clip.events))
    duration_us = max(1.0, end - start)
    playback_seconds = duration_us * 1e-6 / config.replay_speed
    frame_count = max(2, int(math.ceil(playback_seconds * config.replay_fps)))
    frame_timestamps = np.linspace(start, end, frame_count)
    all_events = clip.events

    for timestamp in frame_timestamps:
        if not plt.fignum_exists(figure.number):
            return
        trail_start = timestamp - config.event_trail_us
        frame_events = tuple(
            event for event in all_events if trail_start <= event.t <= timestamp
        )
        frame_clip = HitReplayClip(
            clip.trigger,
            frame_events,
            clip.requested_start_timestamp,
            clip.requested_end_timestamp,
            clip.complete,
        )
        axes.clear()
        _draw_replay_frame(axes, frame_clip, float(timestamp))
        figure.canvas.draw_idle()
        plt.pause(1.0 / config.replay_fps)

    hold_until = time.monotonic() + config.final_hold_sec
    while plt.fignum_exists(figure.number) and time.monotonic() < hold_until:
        plt.pause(min(0.05, max(0.001, hold_until - time.monotonic())))


def _viewer_process_main(replay_queue, config: HitReplayConfig) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[HIT REPLAY] viewer disabled: {error}", flush=True)
        return

    plt.ion()
    figure, axes = plt.subplots(figsize=(6.2, 6.2))
    try:
        figure.canvas.manager.set_window_title("Billiard hit event replay")
    except Exception:
        pass

    while True:
        clip = replay_queue.get()
        if clip is None:
            break
        try:
            _play_clip(figure, axes, clip, config)
        except Exception as error:
            print(f"[HIT REPLAY] clip failed: {error}", flush=True)
    plt.close(figure)


class HitReplayViewer:
    """Small process wrapper; queue overflow drops only the oldest replay."""

    def __init__(self, config: HitReplayConfig = HitReplayConfig()) -> None:
        self.config = config
        self._queue = None
        self._process = None
        self.dropped_clips = 0

    @property
    def alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def start(self) -> bool:
        if not self.config.enabled:
            return False
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=self.config.replay_queue_size)
        self._process = context.Process(
            target=_viewer_process_main,
            args=(self._queue, self.config),
            name="billiard-hit-replay",
            daemon=True,
        )
        self._process.start()
        return True

    def submit(self, clip: HitReplayClip) -> bool:
        if not self.alive or self._queue is None:
            return False
        try:
            self._queue.put_nowait(clip)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.dropped_clips += 1
            except queue.Empty:
                return False
            try:
                self._queue.put_nowait(clip)
                return True
            except queue.Full:
                return False

    def close(self, timeout_sec: float = 2.0) -> None:
        process = self._process
        replay_queue = self._queue
        if process is None:
            return
        if process.is_alive() and replay_queue is not None:
            try:
                replay_queue.put_nowait(None)
            except queue.Full:
                pass
        process.join(timeout=max(0.0, timeout_sec))
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if replay_queue is not None:
            replay_queue.close()
        self._process = None
        self._queue = None
