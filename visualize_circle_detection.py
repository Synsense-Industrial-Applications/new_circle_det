#!/usr/bin/env python3
"""Event-by-event circle detection for 13-point four-region optical flow.

The source CSV stores a 64x64 address and a feature channel. It is decoded to
(x, y, c, t) with the same mapping as ``event_stream_player.py``: the feature
identifies one point in a 2x2 sub-pixel block and one of four diagonal flow
regions. Events are pushed to both detectors one at a time. Playback follows
the source timestamps, and visible events use time-based exponential decay, so
bursts and quiet intervals retain their original timing.

Two detectors are displayed:

* strict DSCT: the original direction-normal formulation;
* adaptive DSCT: upper-arc-aware consensus with confirmed temporal tracking.

The adaptive detector is necessary because this recording's SNN direction
codes have near-random agreement with the radial normals of visible circles.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RangeSlider, Slider, TextBox
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from circle_detection.detector import (  # noqa: E402
    CircleDetection,
    DetectorConfig,
    DirectionalCircleDetector,
    FlowEvent,
)
from circle_detection.adaptive_detector import (  # noqa: E402
    AdaptiveCircleDetection,
    AdaptiveCircleDetector,
    AdaptiveDetectorConfig,
)
from circle_detection.xiaoiron_confidence import (  # noqa: E402
    XiaoironConfig,
    calculate_xiaoiron_confidence,
)
from four_region_flow import FlowData, load_flow_csv  # noqa: E402


DEFAULT_CSV = SCRIPT_DIR / "layer4_20260727_155031_part0001.csv"

# Exact direction convention from event_stream_player.py. Image coordinates
# have +y down, so the vector angles are 45, 135, 225 and 315 degrees.
DIRECTION_NAMES = {0: "down-right", 1: "down-left", 2: "up-left", 3: "up-right"}
DIRECTION_SYMBOLS = {0: "↘", 1: "↙", 2: "↖", 3: "↗"}
DIRECTION_ANGLES_DEG = {0: 45.0, 1: 135.0, 2: 225.0, 3: 315.0}
DIRECTION_VECTORS = {
    0: (0.70710678, 0.70710678),
    1: (-0.70710678, 0.70710678),
    2: (-0.70710678, -0.70710678),
    3: (0.70710678, -0.70710678),
}
DIRECTION_COLORS = {
    0: "#ff453a",
    1: "#ffd60a",
    2: "#0a84ff",
    3: "#30d158",
}
PLAYBACK_SPEED_PRESETS = (
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
CACHE_FORMAT_VERSION = 5
SUPPORT_MODE_CODES = {"unknown": 0, "full": 1, "upper": 2}
SUPPORT_MODE_NAMES = {value: key for key, value in SUPPORT_MODE_CODES.items()}


@dataclass(frozen=True)
class PlaybackCircle:
    cx: float
    cy: float
    radius: float
    confidence: float
    inlier_count: int
    event_count: int
    radial_mad: float
    direction_agreement: float
    upper_angular_sectors: int
    support_mode: str
    timestamp: float


@dataclass(frozen=True)
class CircleSeries:
    cx: np.ndarray
    cy: np.ndarray
    radius: np.ndarray
    confidence: np.ndarray
    inlier_count: np.ndarray
    event_count: np.ndarray
    radial_mad: np.ndarray
    direction_agreement: np.ndarray
    upper_angular_sectors: np.ndarray
    support_mode: np.ndarray
    full_confidence: np.ndarray
    xiaoiron_confidence: np.ndarray
    occlusion_factor: np.ndarray
    direction_factor: np.ndarray
    evidence_strength: np.ndarray
    direction_evidence: np.ndarray
    side_sufficiency: np.ndarray
    side_direction_evidence: np.ndarray
    sector_counts: np.ndarray
    direction_weights: np.ndarray
    sector_visible: np.ndarray

    def at(
        self,
        index: int,
        confidence_threshold: float,
        timestamp: float,
        score_name: str = "confidence",
    ) -> Optional[PlaybackCircle]:
        confidence = float(getattr(self, score_name)[index])
        if not np.isfinite(confidence) or confidence < confidence_threshold:
            return None
        return PlaybackCircle(
            cx=float(self.cx[index]),
            cy=float(self.cy[index]),
            radius=float(self.radius[index]),
            confidence=confidence,
            inlier_count=int(self.inlier_count[index]),
            event_count=int(self.event_count[index]),
            radial_mad=float(self.radial_mad[index]),
            direction_agreement=float(self.direction_agreement[index]),
            upper_angular_sectors=int(self.upper_angular_sectors[index]),
            support_mode=SUPPORT_MODE_NAMES.get(
                int(self.support_mode[index]), "unknown"
            ),
            timestamp=float(timestamp),
        )


@dataclass(frozen=True)
class PrecomputedDetections:
    strict: CircleSeries
    adaptive: CircleSeries

    @property
    def event_count(self) -> int:
        return len(self.adaptive.confidence)


@dataclass(frozen=True)
class XiaoironEvidenceCache:
    """Parameter-independent arrays reused while a formula slider is dragged."""

    counts: np.ndarray
    mass: np.ndarray
    purity: np.ndarray
    sector_visible: np.ndarray


@dataclass(frozen=True)
class ScanResult:
    name: str
    sampled: int
    accepted: int
    best_event_index: Optional[int]
    best_detection: Optional[CircleDetection | AdaptiveCircleDetection | PlaybackCircle]
    median_radius: Optional[float]
    median_direction_agreement: Optional[float]


def make_configs() -> tuple[DetectorConfig, AdaptiveDetectorConfig]:
    """Return original DSCT and the adaptive event-count configuration."""

    strict = replace(DetectorConfig(), max_events=4096)
    adaptive = AdaptiveDetectorConfig()
    return strict, adaptive


def to_flow_events(data: FlowData) -> list[FlowEvent]:
    return [
        FlowEvent(float(x), float(y), int(code), float(timestamp))
        for x, y, code, timestamp in zip(
            data.x,
            data.y,
            data.direction,
            data.timestamp_us,
        )
    ]


def _empty_circle_series(event_count: int) -> CircleSeries:
    nan = lambda: np.full(event_count, np.nan, dtype=np.float32)
    zeros = lambda: np.zeros(event_count, dtype=np.int32)
    return CircleSeries(
        cx=nan(),
        cy=nan(),
        radius=nan(),
        confidence=nan(),
        inlier_count=zeros(),
        event_count=zeros(),
        radial_mad=nan(),
        direction_agreement=nan(),
        upper_angular_sectors=zeros(),
        support_mode=np.zeros(event_count, dtype=np.int8),
        full_confidence=nan(),
        xiaoiron_confidence=nan(),
        occlusion_factor=nan(),
        direction_factor=nan(),
        evidence_strength=nan(),
        direction_evidence=nan(),
        side_sufficiency=np.zeros((event_count, 4), dtype=np.float32),
        side_direction_evidence=np.zeros((event_count, 4), dtype=np.float32),
        sector_counts=np.zeros((event_count, 16), dtype=np.int16),
        direction_weights=np.zeros((event_count, 16, 4), dtype=np.float32),
        sector_visible=np.zeros((event_count, 16), dtype=bool),
    )


def _record_detection(
    series: CircleSeries,
    index: int,
    detection: Optional[CircleDetection | AdaptiveCircleDetection],
) -> None:
    if detection is None:
        return
    series.cx[index] = detection.cx
    series.cy[index] = detection.cy
    series.radius[index] = detection.radius
    series.confidence[index] = detection.confidence
    series.inlier_count[index] = detection.inlier_count
    series.event_count[index] = detection.event_count
    series.radial_mad[index] = detection.radial_mad
    series.direction_agreement[index] = getattr(
        detection, "direction_agreement", np.nan
    )
    series.upper_angular_sectors[index] = getattr(
        detection, "upper_angular_sectors", 0
    )
    series.support_mode[index] = SUPPORT_MODE_CODES.get(
        getattr(detection, "support_mode", "unknown"), 0
    )
    diagnostic = getattr(detection, "xiaoiron", None)
    if diagnostic is not None:
        series.full_confidence[index] = diagnostic.full_confidence
        series.xiaoiron_confidence[index] = diagnostic.xiaoiron_confidence
        series.occlusion_factor[index] = diagnostic.occlusion_factor
        series.direction_factor[index] = diagnostic.direction_factor
        series.evidence_strength[index] = diagnostic.evidence_strength
        series.direction_evidence[index] = diagnostic.direction_evidence
        series.side_sufficiency[index] = diagnostic.side_sufficiency
        series.side_direction_evidence[index] = diagnostic.side_direction_evidence
        series.sector_counts[index] = diagnostic.sector_counts
        series.direction_weights[index] = diagnostic.direction_weights
        series.sector_visible[index] = diagnostic.sector_visible


def build_xiaoiron_evidence_cache(series: CircleSeries) -> XiaoironEvidenceCache:
    hist = np.asarray(series.direction_weights, dtype=float)
    mass = hist.sum(axis=2)
    purity = np.divide(
        hist.max(axis=2),
        mass,
        out=np.zeros_like(mass),
        where=mass > 0.0,
    )
    return XiaoironEvidenceCache(
        counts=np.asarray(series.sector_counts),
        mass=mass,
        purity=purity,
        sector_visible=np.asarray(series.sector_visible),
    )


def recompute_xiaoiron_series(
    series: CircleSeries,
    config: XiaoironConfig,
    cached: Optional[XiaoironEvidenceCache] = None,
) -> None:
    """Recalculate only the confidence formula from cached event evidence.

    Geometry, tracking and event-window extraction are deliberately left
    untouched.  Keeping the complete formula in this one vectorised function
    makes live slider updates fast and makes formula iteration straightforward.

    To try a new formula, edit the five clearly marked lines that compute
    ``side_evidence``, ``direction_evidence``, ``evidence``, ``occlusion`` and
    ``direction`` below.  The player, history curve and display threshold all
    consume the resulting ``series.xiaoiron_confidence`` automatically.
    """

    sides = np.asarray(config.direction_sectors, dtype=int)
    occluded = np.asarray(config.occluded_sectors, dtype=int)
    cached = cached or build_xiaoiron_evidence_cache(series)
    counts = cached.counts
    mass = cached.mass
    purity = cached.purity
    sufficiency = np.minimum(
        1.0,
        np.minimum(
            counts[:, sides] / config.min_direction_events,
            mass[:, sides] / config.min_direction_weight,
        ),
    )

    other = np.ones(16, dtype=bool)
    other[occluded] = False
    other_coverage = np.minimum(
        1.0,
        np.count_nonzero(
            mass[:, other] >= config.min_sector_weight,
            axis=1,
        ) / config.min_other_sectors,
    )
    has_visible_gap = np.any(
        (counts[:, occluded] == 0.0) & cached.sector_visible[:, occluded],
        axis=1,
    )

    # ---- xiaoiron formula: this is the intended iteration point. ----
    side_evidence = sufficiency * purity[:, sides]             # r_s = q_s p_s
    direction_evidence = side_evidence.min(axis=1)  # R = min(r_s), s in {1,2,5,6}
    evidence = np.minimum(sufficiency.min(axis=1), other_coverage)  # G
    occlusion = 1.0 + config.occlusion_gain * has_visible_gap * evidence
    direction = np.exp(-config.direction_penalty_lambda * (1.0 - direction_evidence))
    confidence = np.clip(series.full_confidence * occlusion * direction, 0.0, 1.0)
    # -----------------------------------------------------------------

    valid = np.isfinite(series.full_confidence)
    confidence[~valid] = np.nan
    occlusion[~valid] = np.nan
    direction[~valid] = np.nan
    evidence[~valid] = np.nan
    direction_evidence[~valid] = np.nan
    sufficiency[~valid] = np.nan
    side_evidence[~valid] = np.nan

    series.xiaoiron_confidence[:] = confidence
    series.occlusion_factor[:] = occlusion
    series.direction_factor[:] = direction
    series.evidence_strength[:] = evidence
    series.direction_evidence[:] = direction_evidence
    series.side_sufficiency[:] = sufficiency
    series.side_direction_evidence[:] = side_evidence


def rebuild_xiaoiron_window_evidence(
    data: FlowData,
    series: CircleSeries,
    config: XiaoironConfig,
    *,
    window_events: int,
    decay_events: float,
    radial_tolerance_px: float,
    update_interval_events: int,
) -> np.ndarray:
    """Recount ring evidence for a different recent-event score window.

    The cached centre, radius, ``C_full`` and tracking decisions are retained.
    Only the events used by the xiaoiron score are selected again.  Results are
    held between detector update instants, matching ``AdaptiveCircleDetector``.

    Returns an array containing the actual number of source events inspected at
    each displayed event.  This is separate from ``series.event_count``, which
    describes the cached geometry detector's own window.
    """

    total = len(series.cx)
    if not (
        len(data.x) == len(data.y) == len(data.direction) == total
    ):
        raise ValueError("flow data and circle series must have the same length")
    window_events = max(1, int(window_events))
    update_interval_events = max(1, int(update_interval_events))
    if not np.isfinite(decay_events):
        raise ValueError("decay_events must be finite")
    if not np.isfinite(radial_tolerance_px) or radial_tolerance_px <= 0.0:
        raise ValueError("radial_tolerance_px must be finite and positive")

    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y, dtype=float)
    codes = np.asarray(data.direction, dtype=np.int16)
    age = np.arange(window_events - 1, -1, -1, dtype=float)
    decay_weights = (
        np.exp(-age / decay_events)
        if decay_events > 0.0
        else np.ones(window_events, dtype=float)
    )
    actual_window_counts = np.zeros(total, dtype=np.int32)

    for target in (
        series.xiaoiron_confidence,
        series.occlusion_factor,
        series.direction_factor,
        series.evidence_strength,
        series.direction_evidence,
    ):
        target.fill(np.nan)
    series.side_sufficiency.fill(0.0)
    series.side_direction_evidence.fill(0.0)
    series.sector_counts.fill(0)
    series.direction_weights.fill(0.0)
    series.sector_visible.fill(False)

    for event_index in range(0, total, update_interval_events):
        hold_end = min(total, event_index + update_interval_events)
        hold = slice(event_index, hold_end)
        sample_count = min(window_events, event_index + 1)
        actual_window_counts[hold] = sample_count
        circle_values = (
            series.cx[event_index],
            series.cy[event_index],
            series.radius[event_index],
            series.full_confidence[event_index],
        )
        if not np.all(np.isfinite(circle_values)) or circle_values[2] <= 0.0:
            continue

        begin = event_index + 1 - sample_count
        diagnostic = calculate_xiaoiron_confidence(
            x[begin:event_index + 1],
            y[begin:event_index + 1],
            codes[begin:event_index + 1],
            decay_weights[-sample_count:],
            cx=float(series.cx[event_index]),
            cy=float(series.cy[event_index]),
            radius=float(series.radius[event_index]),
            radial_tolerance_px=float(radial_tolerance_px),
            full_confidence=float(series.full_confidence[event_index]),
            config=config,
        )
        series.sector_counts[hold] = diagnostic.sector_counts
        series.direction_weights[hold] = diagnostic.direction_weights
        series.sector_visible[hold] = diagnostic.sector_visible

    cached = build_xiaoiron_evidence_cache(series)
    recompute_xiaoiron_series(series, config, cached)
    return actual_window_counts


def _print_precompute_progress(done: int, total: int, started: float) -> None:
    fraction = done / max(1, total)
    width = 30
    filled = min(width, int(round(fraction * width)))
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - done) / rate if rate > 0.0 else 0.0
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if done >= total else ""
    print(
        f"\rprecompute [{bar}] {fraction:6.1%}  "
        f"{done:,}/{total:,}  ETA {remaining:5.1f}s",
        end=ending,
        flush=True,
    )


def precompute_detections(
    events: list[FlowEvent],
    strict_config: DetectorConfig,
    adaptive_config: AdaptiveDetectorConfig,
    *,
    show_progress: bool = True,
) -> PrecomputedDetections:
    """Evaluate all source events once and return arrays used during playback."""

    event_count = len(events)
    strict_series = _empty_circle_series(event_count)
    adaptive_series = _empty_circle_series(event_count)
    strict_detector = DirectionalCircleDetector(strict_config, DIRECTION_ANGLES_DEG)
    adaptive_detector = AdaptiveCircleDetector(adaptive_config, DIRECTION_ANGLES_DEG)
    update_interval = max(1, adaptive_config.update_interval_events)
    last_strict: Optional[CircleDetection] = None
    started = time.perf_counter()
    progress_interval = max(1, event_count // 100)

    for index, event in enumerate(events):
        strict_detector.push(event)
        adaptive_detector.push(event)
        if index % update_interval == 0 or index == event_count - 1:
            last_strict = strict_detector.detect(event.t)
        adaptive = adaptive_detector.detect()
        _record_detection(strict_series, index, last_strict)
        _record_detection(adaptive_series, index, adaptive)
        if show_progress and (
            (index + 1) % progress_interval == 0 or index == event_count - 1
        ):
            _print_precompute_progress(index + 1, event_count, started)

    return PrecomputedDetections(strict_series, adaptive_series)


def _cache_signature(
    csv_path: Path,
    event_count: int,
    strict_config: DetectorConfig,
    adaptive_config: AdaptiveDetectorConfig,
) -> str:
    stat = csv_path.stat()
    return json.dumps(
        {
            "format": CACHE_FORMAT_VERSION,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "event_count": event_count,
            "strict": asdict(strict_config),
            "adaptive": asdict(adaptive_config),
            "directions": DIRECTION_ANGLES_DEG,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _series_to_cache(prefix: str, series: CircleSeries) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_{name}": getattr(series, name)
        for name in CircleSeries.__dataclass_fields__
    }


def _series_from_cache(store: np.lib.npyio.NpzFile, prefix: str) -> CircleSeries:
    return CircleSeries(
        **{
            name: np.asarray(store[f"{prefix}_{name}"])
            for name in CircleSeries.__dataclass_fields__
        }
    )


def load_precomputed_cache(
    cache_path: Path,
    signature: str,
    event_count: int,
) -> Optional[PrecomputedDetections]:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as store:
            if str(store["signature"].item()) != signature:
                return None
            result = PrecomputedDetections(
                strict=_series_from_cache(store, "strict"),
                adaptive=_series_from_cache(store, "adaptive"),
            )
        if result.event_count != event_count:
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def save_precomputed_cache(
    cache_path: Path,
    signature: str,
    detections: PrecomputedDetections,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.name}.tmp")
    values: dict[str, object] = {"signature": np.asarray(signature)}
    values.update(_series_to_cache("strict", detections.strict))
    values.update(_series_to_cache("adaptive", detections.adaptive))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(cache_path)


def summarize_series(
    name: str,
    series: CircleSeries,
    events: list[FlowEvent],
    stride: int,
) -> ScanResult:
    stride = max(1, int(stride))
    indices = np.arange(0, len(events), stride, dtype=np.int64)
    if not len(indices) or indices[-1] != len(events) - 1:
        indices = np.append(indices, len(events) - 1)
    confidence = series.confidence[indices]
    valid = np.isfinite(confidence)
    accepted_indices = indices[valid]
    if not len(accepted_indices):
        return ScanResult(name, len(indices), 0, None, None, None, None)
    best_index = int(accepted_indices[np.nanargmax(series.confidence[accepted_indices])])
    best = series.at(best_index, -np.inf, events[best_index].t)
    direction = series.direction_agreement[accepted_indices]
    finite_direction = direction[np.isfinite(direction)]
    return ScanResult(
        name=name,
        sampled=len(indices),
        accepted=len(accepted_indices),
        best_event_index=best_index,
        best_detection=best,
        median_radius=float(np.nanmedian(series.radius[accepted_indices])),
        median_direction_agreement=(
            float(np.median(finite_direction)) if len(finite_direction) else None
        ),
    )


def scan_detector(
    events: list[FlowEvent],
    config: DetectorConfig | AdaptiveDetectorConfig,
    name: str,
    stride: int,
) -> ScanResult:
    """Sample a complete stream without altering event order."""

    is_adaptive = isinstance(config, AdaptiveDetectorConfig)
    detector: DirectionalCircleDetector | AdaptiveCircleDetector
    if is_adaptive:
        detector = AdaptiveCircleDetector(config, DIRECTION_ANGLES_DEG)
    else:
        detector = DirectionalCircleDetector(config, DIRECTION_ANGLES_DEG)
    accepted = 0
    sampled = 0
    best_index: Optional[int] = None
    best: Optional[CircleDetection | AdaptiveCircleDetection] = None
    radii: list[float] = []
    direction_agreements: list[float] = []
    stride = max(1, int(stride))
    for index, event in enumerate(events):
        detector.push(event)
        if index % stride != 0 and index != len(events) - 1:
            continue
        sampled += 1
        if isinstance(detector, AdaptiveCircleDetector):
            detection = detector.detect(force=True)
        else:
            detection = detector.detect(event.t)
        if detection is None:
            continue
        accepted += 1
        radii.append(float(detection.radius))
        if isinstance(detection, AdaptiveCircleDetection):
            direction_agreements.append(float(detection.direction_agreement))
        if best is None or detection.confidence > best.confidence:
            best_index = index
            best = detection
    return ScanResult(
        name,
        sampled,
        accepted,
        best_index,
        best,
        float(np.median(radii)) if radii else None,
        float(np.median(direction_agreements)) if direction_agreements else None,
    )


def print_scan(result: ScanResult, first_timestamp: float) -> None:
    rate = 100.0 * result.accepted / max(1, result.sampled)
    print(
        f"{result.name:>10}: {result.accepted:,}/{result.sampled:,} sampled "
        f"events accepted ({rate:.2f}%)"
    )
    if result.best_detection is None or result.best_event_index is None:
        print("             best: none")
        return
    item = result.best_detection
    relative_s = (item.timestamp - first_timestamp) * 1e-6
    print(
        "             best: "
        f"event={result.best_event_index + 1:,}, t={relative_s:.6f}s, "
        f"center=({item.cx:.2f},{item.cy:.2f}), r={item.radius:.2f}px, "
        f"confidence={item.confidence:.3f}, "
        f"inliers={item.inlier_count}/{item.event_count}, "
        f"radial_MAD={item.radial_mad:.2f}px"
    )
    if result.median_radius is not None:
        detail = f"             median accepted radius: {result.median_radius:.2f}px"
        if result.median_direction_agreement is not None:
            detail += (
                f", direction/radial agreement: "
                f"{result.median_direction_agreement:.3f}"
            )
        print(detail)


class DSCTEventPlayer:
    """Player that evaluates every event and renders at an adjustable speed."""

    def __init__(
        self,
        data: FlowData,
        events: list[FlowEvent],
        precomputed: PrecomputedDetections,
        *,
        trail_ms: float,
        fade_tau_ms: float,
        interval_ms: int,
        playback_speed: float,
        confidence_threshold: float,
        history_events: int,
        start_index: int = 0,
        enable_timer: bool = True,
        confidence_metric: str = "xiaoiron_confidence",
        show_sectors: bool = True,
        xiaoiron_config: Optional[XiaoironConfig] = None,
        score_window_events: int = 300,
        score_decay_events: float = 120.0,
        score_radial_tolerance_px: float = 2.5,
        score_update_interval_events: int = 6,
        display_event_limit: int = 300,
    ) -> None:
        if confidence_metric not in ("confidence", "xiaoiron_confidence"):
            raise ValueError("confidence_metric must be confidence or xiaoiron_confidence")
        if not hasattr(precomputed.adaptive, "xiaoiron_confidence"):
            raise ValueError("Old precomputed results: rerun the Notebook import, parameter and event-loop cells.")
        self.confidence_metric = confidence_metric
        self.show_sectors = show_sectors
        self.xiaoiron_config = xiaoiron_config or XiaoironConfig()
        self.initial_xiaoiron_config = self.xiaoiron_config
        self._updating_xiaoiron_controls = False
        self.data = data
        self.events = events
        if precomputed.event_count != len(events):
            raise ValueError("precomputed detection count does not match event count")
        self.precomputed = precomputed
        self.score_window_events = max(1, int(score_window_events))
        self.initial_score_window_events = self.score_window_events
        self.pending_score_window_events = self.score_window_events
        self.score_decay_events = float(score_decay_events)
        self.score_radial_tolerance_px = float(score_radial_tolerance_px)
        self.score_update_interval_events = max(
            1, int(score_update_interval_events)
        )
        self._score_window_dirty = False
        self.score_window_event_count = np.where(
            np.isfinite(self.precomputed.adaptive.full_confidence),
            np.minimum(
                self.precomputed.adaptive.event_count,
                self.score_window_events,
            ),
            0,
        ).astype(np.int32)
        self.xiaoiron_evidence_cache = build_xiaoiron_evidence_cache(
            self.precomputed.adaptive
        )
        recompute_xiaoiron_series(
            self.precomputed.adaptive,
            self.xiaoiron_config,
            self.xiaoiron_evidence_cache,
        )
        self.trail_us = max(1.0, float(trail_ms)) * 1000.0
        self.fade_tau_us = max(0.1, float(fade_tau_ms)) * 1000.0
        self.display_event_limit = max(10, int(display_event_limit))
        self.recent_event_limit_enabled = False
        self.interval_ms = max(1, int(interval_ms))
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        if playback_speed not in PLAYBACK_SPEED_PRESETS:
            raise ValueError(
                f"playback_speed must be one of {PLAYBACK_SPEED_PRESETS}"
            )
        self.speed_index = PLAYBACK_SPEED_PRESETS.index(float(playback_speed))
        self.playback_speed = PLAYBACK_SPEED_PRESETS[self.speed_index]
        self.history_events = max(100, int(history_events))
        self.timestamps = np.asarray([event.t for event in events], dtype=float)
        if np.any(np.diff(self.timestamps) < 0.0):
            raise ValueError("event timestamps must be sorted in non-decreasing order")
        self.first_timestamp = float(self.timestamps[0])
        self.base_rgba = {
            code: np.asarray(mcolors.to_rgba(color), dtype=float)
            for code, color in DIRECTION_COLORS.items()
        }

        # Reserve a dedicated band above the plots for live event information.
        # Keeping it on a separate axes prevents it from covering the event
        # legend or data when the window is resized.
        self.figure = plt.figure(figsize=(18.0, 10.0))
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.15, 1.0),
            height_ratios=(1.0, 1.0),
            left=0.045,
            right=0.745,
            top=0.69,
            bottom=0.25,
            wspace=0.27,
            hspace=0.38,
        )
        self.ax_events = self.figure.add_subplot(grid[:, 0])
        self.ax_confidence = self.figure.add_subplot(grid[0, 1])
        self.ax_radius = self.figure.add_subplot(grid[1, 1])
        self.ax_info = self.figure.add_axes((0.045, 0.735, 0.700, 0.17))
        self.ax_info.set_axis_off()
        self.ax_sector = self.figure.add_axes((0.485, 0.785, 0.260, 0.12))
        self.ax_formula = self.figure.add_axes((0.765, 0.245, 0.220, 0.665))
        self.ax_formula.set_axis_off()
        self.figure.suptitle(
            "Circle detection - event playback and live confidence tuning",
            fontsize=14,
            y=0.965,
        )

        start_index = int(np.clip(start_index, 0, len(events) - 1))
        self.index = start_index - 1
        self.playhead_timestamp_us = float(self.timestamps[start_index])
        self._clock_anchor_timestamp_us = self.playhead_timestamp_us
        self._clock_anchor_wall_s = time.perf_counter()
        self.loop_enabled = False
        self.loop_start_index = 0
        self.loop_end_index = len(events) - 1
        self.playing = True
        self.animation: Optional[FuncAnimation] = None
        self._create_artists()
        self._create_playback_controls()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect(
            "button_release_event", self._on_mouse_button_release
        )

        self.strict_detection: Optional[PlaybackCircle] = None
        self.adaptive_detection: Optional[PlaybackCircle] = None
        self.strict_confidence = precomputed.strict.confidence
        self.adaptive_confidence = precomputed.adaptive.confidence
        self.strict_radius = precomputed.strict.radius
        self.adaptive_radius = precomputed.adaptive.radius

        if enable_timer:
            self.animation = FuncAnimation(
                self.figure,
                self._timer_tick,
                interval=self.interval_ms,
                cache_frame_data=False,
            )

    def _create_artists(self) -> None:
        self.ax_events.set_xlim(-0.5, 127.5)
        self.ax_events.set_ylim(127.5, -0.5)
        self.ax_events.set_aspect("equal", adjustable="box")
        self.ax_events.set_xlabel("x [pixel]")
        self.ax_events.set_ylabel("y [pixel]")
        self.ax_events.grid(alpha=0.13)

        self.event_scatter = self.ax_events.scatter([], [], s=16, linewidths=0)
        self.current_marker = self.ax_events.scatter(
            [], [], s=90, facecolors="none", edgecolors="#111111", linewidths=1.8, zorder=8
        )
        self.current_arrow = self.ax_events.quiver(
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.007,
            color="#111111",
            zorder=9,
        )

        self.strict_circle = Circle(
            (0.0, 0.0), 1.0, fill=False, edgecolor="#D62728", linewidth=2.4, zorder=7
        )
        self.adaptive_circle = Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            edgecolor="#00A6A6",
            linewidth=2.6,
            zorder=6,
        )
        self.strict_circle.set_visible(False)
        self.adaptive_circle.set_visible(False)
        self.ax_events.add_patch(self.strict_circle)
        self.ax_events.add_patch(self.adaptive_circle)
        (self.strict_center,) = self.ax_events.plot(
            [], [], marker="+", linestyle="none", color="#D62728", markersize=13, markeredgewidth=2.0
        )
        (self.adaptive_center,) = self.ax_events.plot(
            [], [], marker="+", linestyle="none", color="#00A6A6", markersize=12, markeredgewidth=2.0
        )

        direction_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=DIRECTION_COLORS[code],
                label=f"c={code}: {DIRECTION_SYMBOLS[code]} {DIRECTION_NAMES[code]}",
            )
            for code in sorted(DIRECTION_NAMES)
        ]
        direction_handles.extend(
            [
                Line2D([0], [0], color="#D62728", linewidth=2.4, label="strict DSCT"),
                Line2D(
                    [0],
                    [0],
                    color="#00A6A6",
                    linewidth=2.6,
                    label="adaptive DSCT",
                ),
            ]
        )
        self.event_legend = self.ax_events.legend(
            handles=direction_handles, loc="upper right", fontsize=8, ncols=2
        )
        self.adaptive_legend_line = self.event_legend.get_lines()[-1]
        self.adaptive_legend_text = self.event_legend.get_texts()[-1]
        self.sector_lines = [
            self.ax_events.plot([], [], color="#888888", alpha=0.32, linewidth=0.6)[0]
            for _ in range(16)
        ]
        self.sector_labels = [
            self.ax_events.text(
                0, 0, str(s), fontsize=8, ha="center", va="center",
                color=("#B87500" if s in (10, 11, 12, 13) else
                       "#7B2CBF" if s in (1, 2, 5, 6) else "#555555"),
                clip_on=True, visible=False,
            ) for s in range(16)
        ]
        self.ax_sector.set_title("Score window: decayed ring events, sectors 0-15", fontsize=9)
        self.ax_sector.set_xticks(range(16))
        self.ax_sector.tick_params(labelsize=8)
        self.ax_sector.set_ylabel("weight", fontsize=8)
        self.sector_bars = [
            self.ax_sector.bar(np.arange(16), np.zeros(16), color=DIRECTION_COLORS[c],
                               label=f"c={c}") for c in range(4)
        ]
        self.ax_sector.legend(ncols=4, fontsize=7, loc="upper right")
        for s in (1, 2, 5, 6):
            self.ax_sector.axvspan(s-0.45, s+0.45, color="#7B2CBF", alpha=0.12)
        for s in (10, 11, 12, 13):
            self.ax_sector.axvspan(s-0.45, s+0.45, color="#F5B041", alpha=0.18)
        self.info_text = self.ax_info.text(
            0.01,
            0.98,
            "",
            transform=self.ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F7F8FA",
                "alpha": 1.0,
                "edgecolor": "#D5D8DC",
                "linewidth": 0.8,
            },
            zorder=10,
        )
        self.formula_text = self.ax_formula.text(
            0.02,
            0.98,
            "XIAOIRON CONFIDENCE\n\n"
            r"$C_x=clip\{C_{full}(1+\alpha EG)e^{-\lambda(1-R)},0,1\}$"
            "\n\n"
            r"$q_s=min(1,n_s/N_{min},W_s/W_{min})$"
            "\n"
            r"$p_s=max_c(H_{s,c})/\sum_c H_{s,c}$"
            "\n"
            r"$r_s=q_sp_s,\quad R=min(r_1,r_2,r_5,r_6)$"
            "\n"
            r"$E=1$: visible S10/11/12/13 has a gap"
            "\n"
            r"$B=min(1,K_{other}/K_{min})$"
            "\n"
            r"$G=min(q_1,q_2,q_5,q_6,B)$"
            "\n\n"
            "Live controls recalculate score arrays only;\n"
            "circle geometry and tracking stay cached.\n"
            "Formula hook: recompute_xiaoiron_series()",
            transform=self.ax_formula.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#F7F8FA",
                "edgecolor": "#C8CDD2",
                "linewidth": 0.9,
            },
        )
        self.formula_current_text = self.ax_formula.text(
            0.02,
            0.525,
            "Move to a valid detection to inspect components.",
            transform=self.ax_formula.transAxes,
            va="top",
            ha="left",
            fontsize=7.6,
            linespacing=1.15,
            family="monospace",
            color="#333333",
        )

        (self.strict_conf_line,) = self.ax_confidence.plot(
            [], [], color="#D62728", linewidth=1.5, marker=".", markersize=2.5,
            label="strict accepted"
        )
        (self.adaptive_conf_line,) = self.ax_confidence.plot(
            [], [], color="#00A6A6", linewidth=1.4, marker=".", markersize=2.5,
            label="full base"
        )
        (self.upper_conf_line,) = self.ax_confidence.plot(
            [], [], color="#E39126", linewidth=1, label="upper accepted"
        )
        (self.xiaoiron_conf_line,) = self.ax_confidence.plot(
            [], [], color="#7B2CBF", linewidth=1.7, label="xiaoiron"
        )
        self.confidence_threshold_line = self.ax_confidence.axhline(
            self.confidence_threshold,
            color="#555555",
            linewidth=1.2,
            linestyle="--",
            label="display threshold",
        )
        self.ax_confidence.set_ylim(-0.01, 1.01)
        self.ax_confidence.set_title("Confidence before display threshold")
        self.ax_confidence.set_ylabel("confidence")
        self.ax_confidence.grid(alpha=0.17)
        self.ax_confidence.legend(fontsize=8, loc="upper right", ncols=2)

        (self.strict_radius_line,) = self.ax_radius.plot(
            [], [], color="#D62728", linewidth=1.5, marker=".", markersize=2.5,
            label="strict radius"
        )
        (self.adaptive_radius_line,) = self.ax_radius.plot(
            [], [], color="#00A6A6", linewidth=1.4, marker=".", markersize=2.5,
            label="adaptive radius"
        )
        self.ax_radius.set_ylim(0.0, 92.0)
        self.ax_radius.set_title("Estimated radius history")
        self.ax_radius.set_xlabel("event index")
        self.ax_radius.set_ylabel("radius [pixel]")
        self.ax_radius.grid(alpha=0.17)
        self.ax_radius.legend(fontsize=8, loc="upper right")

    def _create_playback_controls(self) -> None:
        """Create controls for progress, threshold and playback speed."""

        progress_ax = self.figure.add_axes((0.12, 0.185, 0.42, 0.022))
        loop_range_ax = self.figure.add_axes((0.12, 0.140, 0.58, 0.022))
        slower_ax = self.figure.add_axes((0.045, 0.042, 0.065, 0.040))
        play_ax = self.figure.add_axes((0.120, 0.042, 0.075, 0.040))
        faster_ax = self.figure.add_axes((0.205, 0.042, 0.065, 0.040))
        loop_ax = self.figure.add_axes((0.280, 0.042, 0.090, 0.040))
        speed_ax = self.figure.add_axes((0.455, 0.085, 0.245, 0.020))
        threshold_ax = self.figure.add_axes((0.455, 0.043, 0.245, 0.020))
        event_jump_ax = self.figure.add_axes((0.655, 0.180, 0.075, 0.033))
        event_jump_button_ax = self.figure.add_axes((0.735, 0.180, 0.035, 0.033))
        trail_mode_ax = self.figure.add_axes((0.810, 0.145, 0.165, 0.033))
        display_count_ax = self.figure.add_axes((0.840, 0.105, 0.100, 0.019))

        self.slower_button = Button(slower_ax, "Slower")
        self.play_button = Button(play_ax, "Pause")
        self.faster_button = Button(faster_ax, "Faster")
        self.loop_button = Button(loop_ax, "Loop: Off")
        self.score_button = Button(
            self.figure.add_axes((0.045, 0.092, 0.195, 0.033)), "Score: xiaoiron"
        )
        self.score_button.label.set_text(
            "Score: xiaoiron" if self.confidence_metric == "xiaoiron_confidence" else "Score: original"
        )
        self.score_button.on_clicked(self._toggle_score_metric)
        self._updating_event_jump = False
        self.event_jump_box = TextBox(
            event_jump_ax,
            "Event",
            initial=f"{int(np.clip(self.index + 2, 1, len(self.events)))}",
        )
        self.event_jump_box.label.set_fontsize(8)
        self.event_jump_box.text_disp.set_fontsize(9)
        self.event_jump_button = Button(event_jump_button_ax, "Go")
        self.event_jump_button.label.set_fontsize(8)
        self.event_jump_box.on_text_change(self._on_event_jump_text_change)
        self.event_jump_box.on_submit(self._on_event_jump_submit)
        self.event_jump_button.on_clicked(
            lambda _event: self._on_event_jump_submit(self.event_jump_box.text)
        )
        self.trail_mode_button = Button(trail_mode_ax, "Trail: time window")
        self.trail_mode_button.label.set_fontsize(8)
        self.trail_mode_button.on_clicked(self._toggle_event_trail_mode)
        self.display_count_slider = Slider(
            display_count_ax,
            "display N",
            10.0,
            5000.0,
            valinit=float(self.display_event_limit),
            valstep=10.0,
            valfmt="%0.0f",
        )
        self.display_count_slider.label.set_fontsize(8)
        self.display_count_slider.valtext.set_fontsize(8)
        self.display_count_slider.on_changed(self._on_display_event_limit)
        self._update_event_trail_controls()
        self._updating_progress = False
        self.progress_slider = Slider(
            progress_ax,
            "Progress",
            1,
            len(self.events),
            valinit=1,
            valstep=1,
        )
        self.loop_slider = RangeSlider(
            loop_range_ax,
            "Loop range",
            1,
            len(self.events),
            valinit=(1, len(self.events)),
            valstep=1,
        )
        self.speed_slider = Slider(
            speed_ax,
            "Time speed",
            0,
            len(PLAYBACK_SPEED_PRESETS) - 1,
            valinit=self.speed_index,
            valstep=1,
        )
        self.threshold_slider = Slider(
            threshold_ax,
            "Min conf",
            0.0,
            1.0,
            valinit=self.confidence_threshold,
            valstep=0.01,
            valfmt="%0.2f",
        )
        self.slower_button.on_clicked(lambda _event: self._change_speed(-1))
        self.play_button.on_clicked(self._toggle_play)
        self.faster_button.on_clicked(lambda _event: self._change_speed(1))
        self.loop_button.on_clicked(self._toggle_loop)
        self.progress_slider.on_changed(self._on_progress_slider)
        self.loop_slider.on_changed(self._on_loop_slider)
        self.speed_slider.on_changed(self._on_speed_slider)
        self.threshold_slider.on_changed(self._on_threshold_slider)
        self._create_xiaoiron_controls()
        self._update_speed_label()
        self._update_progress_label()
        self._update_loop_label()

    def _create_xiaoiron_controls(self) -> None:
        """Create live controls for parameters used only by the score formula."""

        self.score_window_slider = Slider(
            self.figure.add_axes((0.820, 0.485, 0.145, 0.019)),
            "score N",
            20.0,
            1200.0,
            valinit=float(self.score_window_events),
            valstep=10.0,
            valfmt="%0.0f",
        )
        self.score_window_slider.label.set_fontsize(8)
        self.score_window_slider.valtext.set_fontsize(8)
        self.score_window_slider.valtext.set_text(
            f"{self.score_window_events} events"
        )
        self.score_window_slider.on_changed(self._on_score_window_parameter)

        specs = (
            ("occlusion_gain", "alpha", 0.0, 1.0, 0.01, "%0.2f"),
            ("direction_penalty_lambda", "lambda", 0.0, 4.0, 0.05, "%0.2f"),
            ("min_direction_events", "N_min", 1.0, 32.0, 1.0, "%0.0f"),
            ("min_direction_weight", "W_min", 0.5, 16.0, 0.25, "%0.2f"),
            ("min_other_sectors", "K_min", 1.0, 13.0, 1.0, "%0.0f"),
            ("min_sector_weight", "sector W", 0.1, 8.0, 0.1, "%0.1f"),
        )
        slider_y = (0.445, 0.405, 0.365, 0.325, 0.285, 0.245)
        self.xiaoiron_sliders: dict[str, Slider] = {}
        for (name, label, minimum, maximum, step, value_format), y in zip(
            specs, slider_y
        ):
            slider = Slider(
                self.figure.add_axes((0.820, y, 0.145, 0.019)),
                label,
                minimum,
                maximum,
                valinit=float(getattr(self.xiaoiron_config, name)),
                valstep=step,
                valfmt=value_format,
            )
            slider.label.set_fontsize(8)
            slider.valtext.set_fontsize(8)
            slider.on_changed(
                lambda value, parameter=name: self._on_xiaoiron_parameter(
                    parameter, value
                )
            )
            self.xiaoiron_sliders[name] = slider
        self.xiaoiron_reset_button = Button(
            self.figure.add_axes((0.810, 0.195, 0.165, 0.033)),
            "Reset formula params",
        )
        self.xiaoiron_reset_button.label.set_fontsize(8)
        self.xiaoiron_reset_button.on_clicked(self._reset_xiaoiron_parameters)

    def _on_score_window_parameter(self, value: float) -> None:
        self.pending_score_window_events = max(1, int(round(value)))
        self._score_window_dirty = (
            self.pending_score_window_events != self.score_window_events
        )
        suffix = " (release)" if self._score_window_dirty else ""
        self.score_window_slider.valtext.set_text(
            f"{self.pending_score_window_events} events{suffix}"
        )

    def _on_mouse_button_release(self, _event=None) -> None:
        if self._score_window_dirty:
            self._apply_score_window()

    def _apply_score_window(self) -> None:
        """Apply the pending score window and rebuild all ring evidence once."""

        new_window = self.pending_score_window_events
        if new_window == self.score_window_events:
            self._score_window_dirty = False
            return
        was_playing = self.playing
        self._set_playing(False)
        self.score_window_slider.valtext.set_text(f"{new_window} computing...")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        try:
            self.score_window_event_count = rebuild_xiaoiron_window_evidence(
                self.data,
                self.precomputed.adaptive,
                self.xiaoiron_config,
                window_events=new_window,
                decay_events=self.score_decay_events,
                radial_tolerance_px=self.score_radial_tolerance_px,
                update_interval_events=self.score_update_interval_events,
            )
            self.score_window_events = new_window
            self.xiaoiron_evidence_cache = build_xiaoiron_evidence_cache(
                self.precomputed.adaptive
            )
            self._score_window_dirty = False
            self.score_window_slider.valtext.set_text(f"{new_window} events")
            if self.index >= 0:
                self._refresh_current_detections()
                self._render()
        finally:
            if was_playing:
                self._set_playing(True)
            self._reanchor_clock()
        self.figure.canvas.draw_idle()

    def _on_xiaoiron_parameter(self, parameter: str, value: float) -> None:
        if self._updating_xiaoiron_controls:
            return
        integer_parameters = {"min_direction_events", "min_other_sectors"}
        converted = int(round(value)) if parameter in integer_parameters else float(value)
        self.xiaoiron_config = replace(
            self.xiaoiron_config,
            **{parameter: converted},
        )
        recompute_xiaoiron_series(
            self.precomputed.adaptive,
            self.xiaoiron_config,
            self.xiaoiron_evidence_cache,
        )
        if self.index >= 0:
            self._refresh_current_detections()
            self._render()
        self.figure.canvas.draw_idle()

    def _reset_xiaoiron_parameters(self, _event=None) -> None:
        self._updating_xiaoiron_controls = True
        try:
            self.xiaoiron_config = self.initial_xiaoiron_config
            for name, slider in self.xiaoiron_sliders.items():
                slider.set_val(float(getattr(self.xiaoiron_config, name)))
        finally:
            self._updating_xiaoiron_controls = False
        recompute_xiaoiron_series(
            self.precomputed.adaptive,
            self.xiaoiron_config,
            self.xiaoiron_evidence_cache,
        )
        if self.index >= 0:
            self._refresh_current_detections()
            self._render()
        self.figure.canvas.draw_idle()

    def _update_speed_label(self) -> None:
        self.speed_slider.valtext.set_text(f"{self.playback_speed:g}x real time")

    def _clock_target_timestamp_us(self, now_s: Optional[float] = None) -> float:
        """Map elapsed wall time to the source event clock."""

        if now_s is None:
            now_s = time.perf_counter()
        elapsed_s = max(0.0, now_s - self._clock_anchor_wall_s)
        target = (
            self._clock_anchor_timestamp_us
            + elapsed_s * 1_000_000.0 * self.playback_speed
        )
        # Do not clip at the global end here: loop playback needs the amount
        # by which the wall clock passed the selected loop boundary.
        return float(max(target, self.first_timestamp))

    def _wrap_loop_timestamp(self, timestamp_us: float) -> tuple[float, bool]:
        """Wrap a source timestamp into the selected repeat interval."""

        if not self.loop_enabled:
            return timestamp_us, False
        loop_start_us = float(self.timestamps[self.loop_start_index])
        loop_end_us = float(self.timestamps[self.loop_end_index])
        if timestamp_us < loop_start_us:
            return loop_start_us, True
        if timestamp_us <= loop_end_us:
            return timestamp_us, False
        loop_duration_us = loop_end_us - loop_start_us
        if loop_duration_us <= 0.0:
            return loop_start_us, True
        wrapped = loop_start_us + (
            (timestamp_us - loop_start_us) % loop_duration_us
        )
        return wrapped, True

    def _reanchor_clock(self) -> None:
        self._clock_anchor_timestamp_us = self.playhead_timestamp_us
        self._clock_anchor_wall_s = time.perf_counter()

    def _set_playhead(
        self, timestamp_us: float, *, respect_loop: bool = False
    ) -> bool:
        """Advance to the last event whose timestamp is at or before the clock."""

        if respect_loop:
            timestamp_us, _ = self._wrap_loop_timestamp(timestamp_us)
        target = float(np.clip(timestamp_us, self.first_timestamp, self.timestamps[-1]))
        self.playhead_timestamp_us = target
        target_index = int(np.searchsorted(self.timestamps, target, side="right") - 1)
        target_index = int(np.clip(target_index, 0, len(self.events) - 1))
        if respect_loop and self.loop_enabled:
            target_index = int(
                np.clip(
                    target_index, self.loop_start_index, self.loop_end_index
                )
            )
        changed = target_index != self.index
        self.index = target_index
        self._refresh_current_detections()
        return changed

    def _on_speed_slider(self, value: float) -> None:
        if self.playing:
            self._set_playhead(
                self._clock_target_timestamp_us(), respect_loop=True
            )
        self.speed_index = int(
            np.clip(round(value), 0, len(PLAYBACK_SPEED_PRESETS) - 1)
        )
        self.playback_speed = PLAYBACK_SPEED_PRESETS[self.speed_index]
        self._reanchor_clock()
        self._update_speed_label()

    def _update_progress_label(self) -> None:
        shown = max(1, self.index + 1)
        self.progress_slider.valtext.set_text(f"{shown:,}/{len(self.events):,}")

    def _update_loop_label(self) -> None:
        self.loop_slider.valtext.set_text(
            f"{self.loop_start_index + 1:,} - {self.loop_end_index + 1:,}"
        )

    def _on_loop_slider(self, value: tuple[float, float]) -> None:
        start_value, end_value = value
        self.loop_start_index = int(
            np.clip(round(start_value) - 1, 0, len(self.events) - 1)
        )
        self.loop_end_index = int(
            np.clip(round(end_value) - 1, self.loop_start_index, len(self.events) - 1)
        )
        self._update_loop_label()

        if self.loop_enabled and not (
            self.loop_start_index <= self.index <= self.loop_end_index
        ):
            self.index = self.loop_start_index
            self.playhead_timestamp_us = float(self.timestamps[self.index])
            self._refresh_current_detections()
        self._reanchor_clock()
        if self.index >= 0:
            self._render()
        self.figure.canvas.draw_idle()

    def _toggle_loop(self, _event=None) -> None:
        self.loop_enabled = not self.loop_enabled
        self.loop_button.label.set_text(
            "Loop: On" if self.loop_enabled else "Loop: Off"
        )
        if self.loop_enabled and not (
            self.loop_start_index <= self.index <= self.loop_end_index
        ):
            self.index = self.loop_start_index
            self.playhead_timestamp_us = float(self.timestamps[self.index])
            self._refresh_current_detections()
        self._reanchor_clock()
        if self.index >= 0:
            self._render()
        self.figure.canvas.draw_idle()

    def _sync_progress_slider(self) -> None:
        if self.index < 0:
            return
        self._updating_progress = True
        try:
            self.progress_slider.set_val(self.index + 1)
            self._update_progress_label()
        finally:
            self._updating_progress = False

    def _on_progress_slider(self, value: float) -> None:
        if self._updating_progress:
            return
        self._set_playing(False)
        self.index = int(np.clip(round(value) - 1, 0, len(self.events) - 1))
        self.playhead_timestamp_us = float(self.timestamps[self.index])
        self._reanchor_clock()
        self._refresh_current_detections()
        self._set_event_jump_text(self.index + 1)
        self._render()
        self.figure.canvas.draw_idle()

    def _on_event_jump_text_change(self, _text: str) -> None:
        if not self._updating_event_jump:
            self.event_jump_box.text_disp.set_color("#111111")

    def _set_event_jump_text(self, event_number: int) -> None:
        self._updating_event_jump = True
        try:
            self.event_jump_box.set_val(f"{event_number:,}")
            self.event_jump_box.text_disp.set_color("#111111")
        finally:
            self._updating_event_jump = False

    def _on_event_jump_submit(self, text: str) -> None:
        """Pause and jump to an exact one-based event number."""

        if self._updating_event_jump:
            return
        normalized = str(text).strip().replace(",", "")
        try:
            requested = int(normalized)
        except ValueError:
            self.event_jump_box.text_disp.set_color("#C62828")
            self.figure.canvas.draw_idle()
            return

        event_number = int(np.clip(requested, 1, len(self.events)))
        self._set_playing(False)
        self.index = event_number - 1
        self.playhead_timestamp_us = float(self.timestamps[self.index])
        self._reanchor_clock()
        self._refresh_current_detections()
        self._set_event_jump_text(event_number)
        self._render()
        self.figure.canvas.draw_idle()

    def _update_event_trail_controls(self) -> None:
        label = (
            "Trail: last N events"
            if self.recent_event_limit_enabled
            else "Trail: time window"
        )
        self.trail_mode_button.label.set_text(label)
        self.display_count_slider.valtext.set_text(
            f"{self.display_event_limit} events"
        )
        self.display_count_slider.ax.set_alpha(
            1.0 if self.recent_event_limit_enabled else 0.55
        )

    def _toggle_event_trail_mode(self, _event=None) -> None:
        self.recent_event_limit_enabled = not self.recent_event_limit_enabled
        self._update_event_trail_controls()
        if self.index >= 0:
            self._render()
        self.figure.canvas.draw_idle()

    def _on_display_event_limit(self, value: float) -> None:
        self.display_event_limit = max(10, int(round(value)))
        self._update_event_trail_controls()
        if self.recent_event_limit_enabled and self.index >= 0:
            self._render()
        self.figure.canvas.draw_idle()

    def _on_threshold_slider(self, value: float) -> None:
        self.confidence_threshold = float(np.clip(value, 0.0, 1.0))
        self.confidence_threshold_line.set_ydata(
            [self.confidence_threshold, self.confidence_threshold]
        )
        if self.index >= 0:
            self._refresh_current_detections()
            self._render()
            self.figure.canvas.draw_idle()

    def _change_speed(self, amount: int) -> None:
        new_index = int(
            np.clip(self.speed_index + amount, 0, len(PLAYBACK_SPEED_PRESETS) - 1)
        )
        self.speed_slider.set_val(new_index)
        self.figure.canvas.draw_idle()

    def _set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        if self.playing and not playing:
            self._set_playhead(
                self._clock_target_timestamp_us(), respect_loop=True
            )
        elif playing and not self.playing:
            if self.loop_enabled and not (
                self.loop_start_index <= self.index <= self.loop_end_index
            ):
                self.index = self.loop_start_index
                self.playhead_timestamp_us = float(self.timestamps[self.index])
                self._refresh_current_detections()
            self._reanchor_clock()
        self.playing = bool(playing)
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        if self.animation is not None:
            if self.playing:
                self.animation.event_source.start()
            else:
                self.animation.event_source.stop()

    def _toggle_play(self, _event=None) -> None:
        self._set_playing(not self.playing)
        self.figure.canvas.draw_idle()

    def _refresh_current_detections(self) -> None:
        event = self.events[self.index]
        self.strict_detection = self.precomputed.strict.at(
            self.index, self.confidence_threshold, event.t
        )
        self.adaptive_detection = self.precomputed.adaptive.at(
            self.index, self.confidence_threshold, event.t, self.confidence_metric
        )

    def _toggle_score_metric(self, _event=None):
        self.confidence_metric = (
            "confidence" if self.confidence_metric == "xiaoiron_confidence"
            else "xiaoiron_confidence"
        )
        self.score_button.label.set_text(
            "Score: xiaoiron" if self.confidence_metric == "xiaoiron_confidence"
            else "Score: original"
        )
        if self.index >= 0:
            self._refresh_current_detections()
            self._render()
            self.figure.canvas.draw_idle()

    def _render_score_debug(self):
        """Inspect stored scoring evidence, independently of the display threshold."""
        series, i = self.precomputed.adaptive, self.index
        valid = np.isfinite(series.xiaoiron_confidence[i])
        hist = series.direction_weights[i]
        window_count = int(self.score_window_event_count[i])
        ring_count = int(series.sector_counts[i].sum())
        ring_weight = float(hist.sum())
        ring_ratio = ring_count / max(1, window_count)
        self.ax_sector.set_title(
            f"score N={window_count}/{self.score_window_events} | "
            f"ring={ring_count} ({ring_ratio:.1%}) | decayed W={ring_weight:.1f}",
            fontsize=8.5,
        )
        bottom = np.zeros(16)
        for code, bars in enumerate(self.sector_bars):
            for s, bar in enumerate(bars):
                bar.set_y(bottom[s])
                bar.set_height(hist[s, code])
            bottom += hist[:, code]
        self.ax_sector.set_ylim(0, max(1, float(bottom.max())*1.25))
        for s, (line, label) in enumerate(zip(self.sector_lines, self.sector_labels)):
            shown = valid and self.show_sectors
            line.set_visible(shown)
            label.set_visible(shown)
            if shown:
                cx, cy, r = series.cx[i], series.cy[i], series.radius[i]
                a, mid = -np.pi+s*np.pi/8, -np.pi+(s+0.5)*np.pi/8
                line.set_data([cx, cx+r*np.cos(a)], [cy, cy+r*np.sin(a)])
                label.set_position((cx+1.10*r*np.cos(mid), cy+1.10*r*np.sin(mid)))
        if not valid:
            return "xiaoiron   n/a (no accepted geometry)\n"
        direction_sectors = tuple(self.xiaoiron_config.direction_sectors)
        side_stats = []
        for side_index, s in enumerate(direction_sectors):
            total = hist[s].sum()
            p = f"{hist[s].max()/total:.2f}" if total else "n/a"
            side_stats.append(
                f"S{s}:n={series.sector_counts[i,s]},q={series.side_sufficiency[i,side_index]:.2f},"
                f"p={p},r={series.side_direction_evidence[i,side_index]:.2f}"
            )
        occluded_sectors = tuple(self.xiaoiron_config.occluded_sectors)
        n_bottom = ",".join(
            str(series.sector_counts[i, s]) for s in occluded_sectors
        )
        split = max(1, len(side_stats) // 2)
        direction_lines = (
            "dir evidence " + "  ".join(side_stats[:split]) + "\n"
            "             " + "  ".join(side_stats[split:]) + "\n"
        )
        status = "shown" if self.adaptive_detection is not None else "hidden by threshold"
        return (
            f"full base  {series.full_confidence[i]:.3f}   "
            f"xiaoiron {series.xiaoiron_confidence[i]:.3f} ({status})\n"
            f"ring near  {ring_count}/{window_count} events ({ring_ratio:.1%})  "
            f"decayed W={ring_weight:.2f}\n"
            f"factors    occlusion x{series.occlusion_factor[i]:.3f}  "
            f"direction x{series.direction_factor[i]:.3f}  G={series.evidence_strength[i]:.2f}  "
            f"R={series.direction_evidence[i]:.2f}\n"
            f"empty set  {'/'.join(map(str, occluded_sectors))} counts={n_bottom}\n"
            + direction_lines
        )

    def _render_formula_debug(self) -> None:
        """Show the current formula substitution beside the live sliders."""

        series, i = self.precomputed.adaptive, self.index
        if i < 0 or not np.isfinite(series.full_confidence[i]):
            self.formula_current_text.set_text(
                "CURRENT EVENT\nNo accepted circle geometry.\n"
                "Move the progress slider to a valid detection."
            )
            return

        config = self.xiaoiron_config
        sides = np.asarray(config.direction_sectors, dtype=int)
        occluded = np.asarray(config.occluded_sectors, dtype=int)
        counts = series.sector_counts[i].astype(float)
        hist = series.direction_weights[i].astype(float)
        mass = hist.sum(axis=1)
        purity = np.divide(
            hist.max(axis=1),
            mass,
            out=np.zeros(16, dtype=float),
            where=mass > 0.0,
        )
        q = np.minimum(
            1.0,
            np.minimum(
                counts[sides] / config.min_direction_events,
                mass[sides] / config.min_direction_weight,
            ),
        )
        other = np.ones(16, dtype=bool)
        other[occluded] = False
        coverage = min(
            1.0,
            np.count_nonzero(mass[other] >= config.min_sector_weight)
            / config.min_other_sectors,
        )
        empty = [
            int(s) for s in occluded
            if counts[s] == 0 and series.sector_visible[i, s]
        ]
        e_value = int(bool(empty))
        g_value = float(min(float(q.min()), coverage))
        r = q * purity[sides]
        r_value = float(r.min())
        o_value = float(series.occlusion_factor[i])
        d_value = float(series.direction_factor[i])
        full = float(series.full_confidence[i])
        result = float(series.xiaoiron_confidence[i])
        window_count = int(self.score_window_event_count[i])
        ring_count = int(counts.sum())
        ring_weight = float(mass.sum())
        ring_ratio = ring_count / max(1, window_count)
        r_rows = []
        for begin in range(0, len(sides), 2):
            r_rows.append(
                "  ".join(
                    f"S{sides[j]} r={r[j]:.3f}"
                    for j in range(begin, min(begin + 2, len(sides)))
                )
            )
        self.formula_current_text.set_text(
            f"CURRENT EVENT {i + 1:,}\n"
            f"score N={window_count}/{self.score_window_events}  "
            f"ring={ring_count} ({ring_ratio:.1%})  W={ring_weight:.2f}\n"
            + "\n".join(r_rows) + "\n"
            + f"E={e_value} gap={empty or 'none'}  B={coverage:.3f}\n"
            f"G={g_value:.3f}  R={r_value:.3f}\n"
            f"O={o_value:.3f}  D={d_value:.3f}\n"
            f"C=clip({full:.3f} x {o_value:.3f} x {d_value:.3f})={result:.3f}"
        )

    def _evaluate_one(self) -> bool:
        if self.loop_enabled and self.index >= self.loop_end_index:
            self.index = self.loop_start_index
            self.playhead_timestamp_us = float(self.timestamps[self.index])
            self._reanchor_clock()
            self._refresh_current_detections()
            return True
        if self.index + 1 >= len(self.events):
            self._set_playing(False)
            if self.animation is not None:
                self.animation.event_source.stop()
            return False

        self.index += 1
        self.playhead_timestamp_us = float(self.timestamps[self.index])
        self._reanchor_clock()
        self._refresh_current_detections()
        return True

    def _active_slice(self) -> slice:
        if self.recent_event_limit_enabled:
            begin = max(0, self.index + 1 - self.display_event_limit)
            return slice(begin, self.index + 1)
        cutoff_us = self.playhead_timestamp_us - self.trail_us
        begin = int(
            np.searchsorted(
                self.timestamps[: self.index + 1], cutoff_us, side="left"
            )
        )
        return slice(begin, self.index + 1)

    @staticmethod
    def _detection_text(
        prefix: str,
        detection: Optional[CircleDetection | AdaptiveCircleDetection | PlaybackCircle],
    ) -> str:
        if detection is None:
            return f"{prefix:<10} none"
        support_mode = getattr(detection, "support_mode", "unknown")
        mode_suffix = f" [{support_mode}]" if support_mode != "unknown" else ""
        return (
            f"{prefix:<10} ({detection.cx:5.1f},{detection.cy:5.1f}) "
            f"r={detection.radius:5.1f} conf={detection.confidence:.3f} "
            f"in={detection.inlier_count}/{detection.event_count}{mode_suffix}"
        )

    @staticmethod
    def _set_circle_artists(
        circle: Circle,
        center_artist: Line2D,
        detection: Optional[CircleDetection],
    ) -> None:
        visible = detection is not None
        circle.set_visible(visible)
        center_artist.set_visible(visible)
        if detection is not None:
            circle.center = (detection.cx, detection.cy)
            circle.set_radius(detection.radius)
            center_artist.set_data([detection.cx], [detection.cy])

    def _render(self) -> list[object]:
        active = self._active_slice()
        event = self.events[self.index]
        x = self.data.x[active]
        y = self.data.y[active]
        code = self.data.direction[active]
        event_age_us = self.playhead_timestamp_us - self.timestamps[active]
        alpha = np.clip(np.exp(-event_age_us / self.fade_tau_us), 0.015, 1.0)
        if len(x):
            colors = np.vstack(
                [
                    self.base_rgba.get(
                        int(value), np.asarray((0.3, 0.3, 0.3, 1.0))
                    )
                    for value in code
                ]
            )
            colors[:, 3] = alpha
            self.event_scatter.set_offsets(np.column_stack((x, y)))
            self.event_scatter.set_facecolors(colors)
            self.event_scatter.set_edgecolors(colors)
        else:
            self.event_scatter.set_offsets(np.empty((0, 2), dtype=float))

        self.current_marker.set_offsets(np.asarray([[event.x, event.y]]))
        current_event_visible = (
            self.recent_event_limit_enabled
            or self.playhead_timestamp_us - event.t <= self.trail_us
        )
        self.current_marker.set_visible(current_event_visible)
        self.current_arrow.set_visible(current_event_visible)
        vector_x, vector_y = DIRECTION_VECTORS.get(event.c, (0.0, 0.0))
        self.current_arrow.set_offsets(np.asarray([[event.x, event.y]]))
        self.current_arrow.set_UVC(np.asarray([vector_x * 8.0]), np.asarray([vector_y * 8.0]))
        self.current_arrow.set_color(DIRECTION_COLORS.get(event.c, "#111111"))

        self._set_circle_artists(
            self.strict_circle, self.strict_center, self.strict_detection
        )
        self._set_circle_artists(
            self.adaptive_circle, self.adaptive_center, self.adaptive_detection
        )
        self.adaptive_circle.set_edgecolor(
            "#7B2CBF" if self.confidence_metric == "xiaoiron_confidence" else "#00A6A6"
        )
        self.adaptive_center.set_color(self.adaptive_circle.get_edgecolor())
        self.adaptive_legend_line.set_color(self.adaptive_circle.get_edgecolor())
        self.adaptive_legend_text.set_text(
            "xiaoiron circle" if self.confidence_metric == "xiaoiron_confidence"
            else "original circle"
        )
        debug_text = self._render_score_debug()
        self._render_formula_debug()

        relative_s = (self.playhead_timestamp_us - self.first_timestamp) * 1e-6
        trail_status = (
            f"last N={self.display_event_limit}"
            if self.recent_event_limit_enabled
            else f"time window={self.trail_us / 1000.0:g} ms"
        )
        self.ax_events.set_title(
            f"Event view  t={relative_s:.6f} s | visible={len(x)} | "
            f"{trail_status} | timestamp fade",
            fontsize=10,
        )
        direction_agreement = (
            f"{self.adaptive_detection.direction_agreement:.3f}"
            if self.adaptive_detection is not None
            else "n/a"
        )
        loop_status = (
            f"on {self.loop_start_index + 1:,}-{self.loop_end_index + 1:,}"
            if self.loop_enabled
            else "off"
        )
        self.info_text.set_text(
            f"event      {self.index + 1:,}/{len(self.events):,}\n"
            f"current    ({event.x:3.0f},{event.y:3.0f}) c={event.c} "
            f"{DIRECTION_NAMES.get(event.c, '?')}\n"
            f"strict     {self._detection_text('', self.strict_detection).strip()}\n"
            f"adaptive   {self._detection_text('', self.adaptive_detection).strip()}\n"
            + debug_text
            + f"dir agree  {direction_agreement}\n"
            f"playback   {self.playback_speed:g}x real time  "
            f"frame={self.interval_ms} ms  loop={loop_status}\n"
            f"min conf   {self.confidence_threshold:.2f}"
            + "\nSpace play | Right step | R restart | +/- speed | S sectors"
        )

        left = max(0, self.index - self.history_events + 1)
        indices = np.arange(left, self.index + 1) + 1
        self.strict_conf_line.set_data(indices, self.strict_confidence[left : self.index + 1])
        self.adaptive_conf_line.set_data(
            indices, self.precomputed.adaptive.full_confidence[left : self.index + 1]
        )
        mode = self.precomputed.adaptive.support_mode[left : self.index + 1]
        self.upper_conf_line.set_data(
            indices, np.where(mode == 2, self.adaptive_confidence[left:self.index+1], np.nan)
        )
        self.xiaoiron_conf_line.set_data(
            indices, self.precomputed.adaptive.xiaoiron_confidence[left:self.index+1]
        )
        self.strict_radius_line.set_data(indices, self.strict_radius[left : self.index + 1])
        self.adaptive_radius_line.set_data(
            indices, self.adaptive_radius[left : self.index + 1]
        )
        history_right = max(self.history_events, self.index + 20)
        self.ax_confidence.set_xlim(max(1, history_right - self.history_events), history_right)
        self.ax_radius.set_xlim(max(1, history_right - self.history_events), history_right)
        self._sync_progress_slider()
        return [
            self.event_scatter,
            self.current_marker,
            self.current_arrow,
            self.strict_circle,
            self.adaptive_circle,
            self.strict_center,
            self.adaptive_center,
            self.strict_conf_line,
            self.adaptive_conf_line,
            self.strict_radius_line,
            self.adaptive_radius_line,
            self.confidence_threshold_line,
            self.info_text,
        ]

    def step(self) -> list[object]:
        if not self._evaluate_one():
            return []
        return self._render()

    def _timer_tick(self, _frame_number: int) -> list[object]:
        if not self.playing:
            return []
        # The wall clock determines the source timestamp to display. Dense
        # bursts therefore contribute many events in one rendered frame,
        # while timestamp gaps remain visible as real pauses. Rendering still
        # runs during gaps so old events continue fading with elapsed time.
        target_timestamp_us = self._clock_target_timestamp_us()

        target_timestamp_us, wrapped = self._wrap_loop_timestamp(
            target_timestamp_us
        )
        self._set_playhead(target_timestamp_us, respect_loop=True)
        if wrapped:
            self._reanchor_clock()

        artists = self._render()
        if not self.loop_enabled and self.playhead_timestamp_us >= self.timestamps[-1]:
            self._set_playing(False)
        return artists

    def _on_key(self, event) -> None:
        if event.key and event.key.lower() == "s":
            self.show_sectors = not self.show_sectors
            if self.index >= 0:
                self._render()
                self.figure.canvas.draw_idle()
            return
        if event.key == " ":
            self._toggle_play()
            return
        if event.key == "right":
            self._set_playing(False)
            self.step()
            self.figure.canvas.draw_idle()
            return
        if event.key and event.key.lower() == "r":
            self.restart()
            return
        if event.key in ("+", "="):
            self._change_speed(1)
        elif event.key in ("-", "_"):
            self._change_speed(-1)
        else:
            return

    def restart(self) -> None:
        self.index = self.loop_start_index if self.loop_enabled else 0
        self.playhead_timestamp_us = float(self.timestamps[self.index])
        self._refresh_current_detections()
        self._set_playing(True)
        self._reanchor_clock()
        self._render()
        self.figure.canvas.draw_idle()

    def seek_for_preview(self, target_index: int, detection_stride: int = 10) -> None:
        """Jump directly to a precomputed event for a static preview."""

        _ = detection_stride
        self.index = int(np.clip(target_index, 0, len(self.events) - 1))
        self.playhead_timestamp_us = float(self.timestamps[self.index])
        self._reanchor_clock()
        self._refresh_current_detections()
        self._render()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Play 13-point four-region optical-flow events through circle detection one event at a time"
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=40,
        help="wall-clock delay per rendered frame; 40 ms is 25 FPS",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        choices=PLAYBACK_SPEED_PRESETS,
        default=1.0,
        help="source-time multiplier; 1.0 preserves the recorded timing",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.30,
        help="initial display threshold for both detected circles",
    )
    parser.add_argument(
        "--trail-ms", type=float, default=160.0, help="visible source-time window in ms"
    )
    parser.add_argument(
        "--fade-tau-ms", type=float, default=50.0, help="opacity decay constant in source-time ms"
    )
    parser.add_argument("--history-events", type=int, default=1500)
    parser.add_argument(
        "--start-event",
        type=int,
        default=1,
        help="one-based event number at which interactive playback starts",
    )
    parser.add_argument("--analyze", action="store_true", help="compare original and adaptive DSCT")
    parser.add_argument("--analysis-stride", type=int, default=10)
    parser.add_argument("--save-preview", type=Path, help="write one PNG instead of only opening the GUI")
    parser.add_argument(
        "--preview-event",
        type=int,
        help="one-based preview event; default is the best sampled adaptive detection",
    )
    parser.add_argument("--save-gif", type=Path, help="write an event-by-event GIF")
    parser.add_argument("--gif-start", type=int, help="one-based first GIF event")
    parser.add_argument("--gif-events", type=int, default=180)
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="ignore and rebuild the precomputed detection cache",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR / path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = load_flow_csv(csv_path)
    events = to_flow_events(data)
    strict_config, adaptive_config = make_configs()
    print(f"source:   {csv_path}")
    print(f"events:   {len(events):,}")
    print(f"duration: {data.duration_s:.6f} s")
    print("mapping:  0=down-right, 1=down-left, 2=up-left, 3=up-right")
    print(f"playback: timestamp driven, {args.playback_speed:g}x real time")

    cache_path = csv_path.with_name(
        f"{csv_path.stem}.circle_cache_v{CACHE_FORMAT_VERSION}.npz"
    )
    signature = _cache_signature(
        csv_path, len(events), strict_config, adaptive_config
    )
    precomputed = None if args.rebuild_cache else load_precomputed_cache(
        cache_path, signature, len(events)
    )
    if precomputed is not None:
        print(f"cache:    loaded {cache_path}")
    else:
        print("cache:    computing all detections before playback")
        precomputed = precompute_detections(
            events, strict_config, adaptive_config, show_progress=True
        )
        save_precomputed_cache(cache_path, signature, precomputed)
        print(f"cache:    saved {cache_path}")

    strict_scan: Optional[ScanResult] = None
    adaptive_scan: Optional[ScanResult] = None
    needs_scan = args.analyze or (
        args.save_preview is not None and args.preview_event is None
    ) or (args.save_gif is not None and args.gif_start is None)
    if needs_scan:
        strict_scan = summarize_series(
            "strict", precomputed.strict, events, args.analysis_stride
        )
        adaptive_scan = summarize_series(
            "adaptive", precomputed.adaptive, events, args.analysis_stride
        )
        print_scan(strict_scan, events[0].t)
        print_scan(adaptive_scan, events[0].t)

    if args.analyze and args.no_show and args.save_preview is None and args.save_gif is None:
        return 0

    if args.save_preview is not None:
        if args.preview_event is not None:
            preview_index = int(np.clip(args.preview_event - 1, 0, len(events) - 1))
        elif adaptive_scan is not None and adaptive_scan.best_event_index is not None:
            preview_index = adaptive_scan.best_event_index
        else:
            preview_index = len(events) - 1
        player = DSCTEventPlayer(
            data,
            events,
            precomputed,
            trail_ms=args.trail_ms,
            fade_tau_ms=args.fade_tau_ms,
            interval_ms=args.interval_ms,
            playback_speed=1.0,
            confidence_threshold=args.confidence_threshold,
            history_events=args.history_events,
            enable_timer=False,
            xiaoiron_config=adaptive_config.xiaoiron,
            score_window_events=adaptive_config.window_events,
            score_decay_events=adaptive_config.decay_events,
            score_radial_tolerance_px=adaptive_config.radial_tolerance_px,
            score_update_interval_events=adaptive_config.update_interval_events,
        )
        player.seek_for_preview(preview_index, args.analysis_stride)
        output = resolve_output(args.save_preview)
        output.parent.mkdir(parents=True, exist_ok=True)
        player.figure.savefig(output, dpi=150, bbox_inches="tight")
        print(f"preview:  {output.resolve()}")
        plt.close(player.figure)

    if args.save_gif is not None:
        if args.gif_start is not None:
            gif_start = int(np.clip(args.gif_start - 1, 0, len(events) - 1))
        elif adaptive_scan is not None and adaptive_scan.best_event_index is not None:
            gif_start = max(0, adaptive_scan.best_event_index - args.gif_events // 2)
        else:
            gif_start = 0
        gif_count = min(max(1, args.gif_events), len(events) - gif_start)
        player = DSCTEventPlayer(
            data,
            events,
            precomputed,
            trail_ms=args.trail_ms,
            fade_tau_ms=args.fade_tau_ms,
            interval_ms=max(1, 1000 // max(1, args.gif_fps)),
            playback_speed=1.0,
            confidence_threshold=args.confidence_threshold,
            history_events=args.history_events,
            start_index=gif_start,
            enable_timer=False,
            xiaoiron_config=adaptive_config.xiaoiron,
            score_window_events=adaptive_config.window_events,
            score_decay_events=adaptive_config.decay_events,
            score_radial_tolerance_px=adaptive_config.radial_tolerance_px,
            score_update_interval_events=adaptive_config.update_interval_events,
        )
        animation = FuncAnimation(
            player.figure,
            lambda _frame: player.step(),
            frames=gif_count,
            init_func=lambda: [],
            interval=max(1, 1000 // max(1, args.gif_fps)),
            cache_frame_data=False,
        )
        output = resolve_output(args.save_gif)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=PillowWriter(fps=max(1, args.gif_fps)), dpi=82)
        print(f"gif:      {output.resolve()}")
        plt.close(player.figure)

    if not args.no_show:
        start_index = int(np.clip(args.start_event - 1, 0, len(events) - 1))
        player = DSCTEventPlayer(
            data,
            events,
            precomputed,
            trail_ms=args.trail_ms,
            fade_tau_ms=args.fade_tau_ms,
            interval_ms=args.interval_ms,
            playback_speed=args.playback_speed,
            confidence_threshold=args.confidence_threshold,
            history_events=args.history_events,
            start_index=start_index,
            enable_timer=True,
            xiaoiron_config=adaptive_config.xiaoiron,
            score_window_events=adaptive_config.window_events,
            score_decay_events=adaptive_config.decay_events,
            score_radial_tolerance_px=adaptive_config.radial_tolerance_px,
            score_update_interval_events=adaptive_config.update_interval_events,
        )
        # Keep a live reference until the GUI closes.
        _ = player
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
