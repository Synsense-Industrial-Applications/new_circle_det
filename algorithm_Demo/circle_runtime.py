"""Hardware-independent real-time circle detection pipeline.

The hardware demo only has to feed raw Layer-4 events into
``CircleDetectionPipeline``.  This module owns event decoding, the latest
adaptive detector, output filtering and runtime counters so those pieces can
be tested without a Speck2f board or samna.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import time
from typing import Callable, Optional, Sequence

from circle_detection import (
    AdaptiveCircleDetection,
    AdaptiveCircleDetector,
    AdaptiveDetectorConfig,
    FlowEvent,
)


# Layer-4 feature -> four diagonal optical-flow directions.  This is the same
# mapping used by four_region_flow.py and the offline latest-method player.
FEATURE_TO_DIRECTION = (0, 1, 1, 0, 2, 3, 3, 2)
DIRECTION_ANGLES_DEG = {0: 45.0, 1: 135.0, 2: 225.0, 3: 315.0}


@dataclass(frozen=True)
class CircleFilterConfig:
    """All output-filter parameters in one place.

    Radius uses a closed interval; centre X/Y use open intervals because the
    requested ranges were written as ``(30, 90)`` and ``(40, 100)``.
    """

    confidence_attribute: str = "xiaoiron_confidence"
    min_confidence: float = 0.30
    min_radius_px: float = 35.0
    max_radius_px: float = 41.0
    min_center_x_px: float = 30.0
    max_center_x_px: float = 90.0
    min_center_y_px: float = 40.0
    max_center_y_px: float = 100.0

    def __post_init__(self) -> None:
        numeric = (
            self.min_confidence,
            self.min_radius_px,
            self.max_radius_px,
            self.min_center_x_px,
            self.max_center_x_px,
            self.min_center_y_px,
            self.max_center_y_px,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("circle filter values must be finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_radius_px <= 0.0 or self.min_radius_px > self.max_radius_px:
            raise ValueError("radius range must be positive and ordered")
        if self.min_center_x_px >= self.max_center_x_px:
            raise ValueError("center X range must be ordered")
        if self.min_center_y_px >= self.max_center_y_px:
            raise ValueError("center Y range must be ordered")
        if not self.confidence_attribute:
            raise ValueError("confidence_attribute must not be empty")


@dataclass(frozen=True)
class FilterDecision:
    """Result of one named output-filter rule."""

    rule_name: str
    passed: bool
    actual: str
    expected: str

    @property
    def message(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.rule_name}={status} ({self.actual}; expected {self.expected})"


@dataclass(frozen=True)
class FilterReport:
    """Complete report; all rules run so the terminal can show every cause."""

    decisions: tuple[FilterDecision, ...]

    @property
    def passed(self) -> bool:
        return all(decision.passed for decision in self.decisions)

    @property
    def failures(self) -> tuple[FilterDecision, ...]:
        return tuple(decision for decision in self.decisions if not decision.passed)

    def failure_summary(self) -> str:
        return "; ".join(decision.message for decision in self.failures)


FilterEvaluator = Callable[
    [AdaptiveCircleDetection, CircleFilterConfig],
    FilterDecision,
]


@dataclass(frozen=True)
class CircleFilterRule:
    """A registered filter rule.

    To add a filter, write one small evaluator returning ``FilterDecision`` and
    append ``CircleFilterRule(name, evaluator)`` to the registry used by the
    demo.  The hardware loop and statistics code do not need to change.
    """

    name: str
    evaluator: FilterEvaluator

    def evaluate(
        self,
        detection: AdaptiveCircleDetection,
        config: CircleFilterConfig,
    ) -> FilterDecision:
        decision = self.evaluator(detection, config)
        if decision.rule_name != self.name:
            raise ValueError(
                f"filter {self.name!r} returned rule name {decision.rule_name!r}"
            )
        return decision


def _confidence_filter(
    detection: AdaptiveCircleDetection,
    config: CircleFilterConfig,
) -> FilterDecision:
    try:
        confidence = float(getattr(detection, config.confidence_attribute))
    except (AttributeError, TypeError, ValueError):
        confidence = math.nan
    passed = math.isfinite(confidence) and confidence >= config.min_confidence
    return FilterDecision(
        "confidence",
        passed,
        f"{config.confidence_attribute}={confidence:.3f}",
        f">= {config.min_confidence:.3f}",
    )


def _radius_filter(
    detection: AdaptiveCircleDetection,
    config: CircleFilterConfig,
) -> FilterDecision:
    radius = float(detection.radius)
    passed = (
        math.isfinite(radius)
        and config.min_radius_px <= radius <= config.max_radius_px
    )
    return FilterDecision(
        "radius",
        passed,
        f"r={radius:.3f}px",
        f"[{config.min_radius_px:g}, {config.max_radius_px:g}]px",
    )


def _center_x_filter(
    detection: AdaptiveCircleDetection,
    config: CircleFilterConfig,
) -> FilterDecision:
    center_x = float(detection.cx)
    passed = (
        math.isfinite(center_x)
        and config.min_center_x_px < center_x < config.max_center_x_px
    )
    return FilterDecision(
        "center_x",
        passed,
        f"cx={center_x:.3f}px",
        f"({config.min_center_x_px:g}, {config.max_center_x_px:g})px",
    )


def _center_y_filter(
    detection: AdaptiveCircleDetection,
    config: CircleFilterConfig,
) -> FilterDecision:
    center_y = float(detection.cy)
    passed = (
        math.isfinite(center_y)
        and config.min_center_y_px < center_y < config.max_center_y_px
    )
    return FilterDecision(
        "center_y",
        passed,
        f"cy={center_y:.3f}px",
        f"({config.min_center_y_px:g}, {config.max_center_y_px:g})px",
    )


# Evaluation order is explicit and stable.  New conditions only need a named
# evaluator plus one registry entry here (or in Demo.py's custom tuple).
DEFAULT_CIRCLE_FILTERS: tuple[CircleFilterRule, ...] = (
    CircleFilterRule("confidence", _confidence_filter),
    CircleFilterRule("radius", _radius_filter),
    CircleFilterRule("center_x", _center_x_filter),
    CircleFilterRule("center_y", _center_y_filter),
)


class DetectionFilterChain:
    """Evaluate a configurable sequence of independent output rules."""

    def __init__(
        self,
        config: CircleFilterConfig,
        rules: Sequence[CircleFilterRule] = DEFAULT_CIRCLE_FILTERS,
    ) -> None:
        if not rules:
            raise ValueError("at least one circle filter rule is required")
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)):
            raise ValueError("circle filter rule names must be unique")
        self.config = config
        self.rules = tuple(rules)

    def evaluate(self, detection: AdaptiveCircleDetection) -> FilterReport:
        return FilterReport(
            tuple(rule.evaluate(detection, self.config) for rule in self.rules)
        )


def decode_layer4_event(
    x64: int,
    y64: int,
    feature: int,
    timestamp: float,
) -> FlowEvent:
    """Decode one raw 64x64x8 Layer-4 event to ``(x128, y128, c, t)``."""

    x64 = int(x64)
    y64 = int(y64)
    feature = int(feature)
    if not 0 <= x64 < 64 or not 0 <= y64 < 64:
        raise ValueError(f"Layer-4 coordinates out of range: ({x64}, {y64})")
    if not 0 <= feature < len(FEATURE_TO_DIRECTION):
        raise ValueError(f"Layer-4 feature out of range: {feature}")
    feature_mod4 = feature % 4
    x128 = 2 * x64 + feature_mod4 // 2
    y128 = 2 * y64 + feature_mod4 % 2
    direction = FEATURE_TO_DIRECTION[feature]
    return FlowEvent(float(x128), float(y128), direction, float(timestamp))


@dataclass(frozen=True)
class CirclePipelineUpdate:
    """One real detector update (never a repeated cached result)."""

    update_number: int
    source_event_count: int
    event: FlowEvent
    detection: Optional[AdaptiveCircleDetection]
    filter_report: Optional[FilterReport]
    detection_ms: float

    @property
    def accepted(self) -> bool:
        return bool(self.filter_report and self.filter_report.passed)


@dataclass
class CircleRuntimeStats:
    """Cumulative counters used by the terminal reporter."""

    source_events: int = 0
    invalid_events: int = 0
    detector_updates: int = 0
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    no_candidate: int = 0
    rejection_counts: Counter[str] = field(default_factory=Counter)
    recent_detection_times_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    total_detection_ms: float = 0.0
    overall_max_detection_ms: float = 0.0

    def record_update(self, update: CirclePipelineUpdate) -> None:
        self.detector_updates += 1
        self.recent_detection_times_ms.append(update.detection_ms)
        self.total_detection_ms += update.detection_ms
        self.overall_max_detection_ms = max(
            self.overall_max_detection_ms,
            update.detection_ms,
        )
        if update.detection is None:
            self.no_candidate += 1
            return
        self.candidates += 1
        if update.accepted:
            self.accepted += 1
            return
        self.rejected += 1
        if update.filter_report is not None:
            for failure in update.filter_report.failures:
                self.rejection_counts[failure.rule_name] += 1

    @property
    def average_detection_ms(self) -> float:
        if not self.detector_updates:
            return 0.0
        return self.total_detection_ms / self.detector_updates

    @property
    def p95_detection_ms(self) -> float:
        if not self.recent_detection_times_ms:
            return 0.0
        ordered = sorted(self.recent_detection_times_ms)
        return ordered[math.ceil(0.95 * len(ordered)) - 1]

    @property
    def max_detection_ms(self) -> float:
        return self.overall_max_detection_ms


class CircleDetectionPipeline:
    """Long-lived latest-method detector plus extensible output filters."""

    def __init__(
        self,
        detector_config: AdaptiveDetectorConfig,
        filter_config: CircleFilterConfig,
        filter_rules: Sequence[CircleFilterRule] = DEFAULT_CIRCLE_FILTERS,
        *,
        detector: Optional[AdaptiveCircleDetector] = None,
    ) -> None:
        self.detector_config = detector_config
        self.filter_chain = DetectionFilterChain(filter_config, filter_rules)
        self.detector = detector or AdaptiveCircleDetector(
            detector_config,
            DIRECTION_ANGLES_DEG,
        )
        self.stats = CircleRuntimeStats()
        self._update_number = 0

    def reset(self) -> None:
        self.detector.reset()
        self.stats = CircleRuntimeStats()
        self._update_number = 0

    def process_raw_event(
        self,
        x64: int,
        y64: int,
        feature: int,
        timestamp: float,
    ) -> Optional[CirclePipelineUpdate]:
        """Decode one raw Layer-4 event and pass it to the shared pipeline."""

        try:
            event = decode_layer4_event(x64, y64, feature, timestamp)
        except (TypeError, ValueError):
            self.stats.invalid_events += 1
            raise

        return self.process_flow_event(event)

    def process_flow_event(
        self,
        event: FlowEvent,
    ) -> Optional[CirclePipelineUpdate]:
        """Consume one decoded event and return only on a real update.

        Every event is retained in arrival order.  ``detect()`` runs only at the
        configured update instants, so downstream output and counters never
        duplicate one cached circle.  This entry point also enables CSV replay
        and tests without importing the hardware demo.
        """

        self.detector.push(event)
        self.stats.source_events += 1
        event_index = self.stats.source_events - 1
        update_interval = max(1, self.detector_config.update_interval_events)
        is_update_event = event_index % update_interval == 0

        if not is_update_event:
            return None

        # We own the exact update schedule, so skip even the detector method
        # call on the other events.  ``force=True`` makes this schedule explicit
        # and still matches the detector's native 0, interval, 2*interval, ...
        # event indices.
        started = time.perf_counter()
        detection = self.detector.detect(force=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._update_number += 1
        report = (
            self.filter_chain.evaluate(detection)
            if detection is not None
            else None
        )
        update = CirclePipelineUpdate(
            update_number=self._update_number,
            source_event_count=self.stats.source_events,
            event=event,
            detection=detection,
            filter_report=report,
            detection_ms=elapsed_ms,
        )
        self.stats.record_update(update)
        return update
