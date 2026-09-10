"""Adaptive event-count circle consensus for non-radial SNN flow codes.

The original DSCT detector is very efficient when ``c`` is a quantised local
circle normal.  Some SNN flow layers instead encode temporal motion direction;
on those recordings the direction code is not a valid centre-line constraint.

This module keeps DSCT's streaming interface, but obtains radius-free circle
hypotheses from random triples of recent event positions. Three non-collinear
points define one circle, so there is still no radius sweep or 3-D Hough
accumulator. A dedicated upper-semicircle validator handles cue-mounted camera
occlusion. A confirmed temporal track rejects isolated centre/radius jumps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Deque, Iterable, Mapping, Optional

import numpy as np

from .detector import FlowEvent
from .xiaoiron_confidence import (
    XiaoironConfig, XiaoironScore, calculate_xiaoiron_confidence,
)


@dataclass(frozen=True)
class AdaptiveDetectorConfig:
    width: int = 128
    height: int = 128
    window_events: int = 300
    decay_events: float = 120.0
    min_events: int = 80
    hypotheses: int = 64
    refine_candidates: int = 10
    refine_iterations: int = 4
    update_interval_events: int = 6
    min_radius_px: float = 4.0
    max_radius_px: float = 90.0
    centre_margin_px: float = 8.0
    radial_tolerance_px: float = 2.5
    max_radial_mad_px: float = 1.8
    min_inliers: int = 36
    min_inlier_weight_ratio: float = 0.24
    angular_sector_count: int = 16
    min_angular_sector_weight: float = 1.0
    min_angular_sectors: int = 7
    min_quadrants: int = 3
    allow_upper_arc: bool = True
    upper_min_inliers: int = 28
    upper_min_inlier_weight_ratio: float = 0.18
    upper_min_angular_sectors: int = 5
    upper_min_span_deg: float = 125.0
    upper_endpoint_radius_ratio: float = 0.32
    upper_apex_radius_ratio: float = 0.58
    upper_confidence_scale: float = 1.12
    min_direction_agreement: float = 0.05
    min_confidence: float = 0.18
    upper_min_confidence: float = 0.14
    continuation_inlier_scale: float = 0.80
    continuation_ratio_scale: float = 0.85
    continuation_sector_relaxation: int = 1
    continuation_confidence_scale: float = 0.80
    tracking_bonus: float = 0.10
    smoothing: float = 0.22
    max_center_jump_px: float = 7.0
    max_center_jump_radius_ratio: float = 0.05
    max_radius_jump_px: float = 4.5
    max_radius_jump_ratio: float = 0.05
    jump_confirmations: int = 3
    track_reset_updates: int = 3
    pending_center_tolerance_px: float = 6.0
    pending_radius_tolerance_px: float = 4.0
    seed: int = 20260907
    xiaoiron: XiaoironConfig = field(default_factory=XiaoironConfig)


@dataclass(frozen=True)
class AdaptiveCircleDetection:
    cx: float
    cy: float
    radius: float
    confidence: float
    event_count: int
    inlier_count: int
    radial_inlier_ratio: float
    radial_mad: float
    angular_sectors: int
    quadrants: int
    upper_angular_sectors: int
    support_mode: str
    direction_agreement: float
    hypotheses_tested: int
    timestamp: float
    event_index: int
    xiaoiron: Optional[XiaoironScore] = None

    @property
    def xiaoiron_confidence(self) -> float:
        return self.xiaoiron.xiaoiron_confidence if self.xiaoiron else math.nan

    @property
    def full_confidence(self) -> float:
        return self.xiaoiron.full_confidence if self.xiaoiron else math.nan


@dataclass(frozen=True)
class _ScoredCircle:
    cx: float
    cy: float
    radius: float
    confidence: float
    inlier_count: int
    radial_ratio: float
    radial_mad: float
    angular_sectors: int
    quadrants: int
    upper_angular_sectors: int
    support_mode: str
    direction_agreement: float


class AdaptiveCircleDetector:
    """Streaming circle detector with a fixed event-count memory.

    Unlike time-window DSCT, the amount of geometric evidence is stable when
    the event rate changes sharply.  Calling :meth:`detect` more frequently
    than ``update_interval_events`` returns the last estimate, while every
    input event is still retained in exact arrival order.
    """

    def __init__(
        self,
        config: AdaptiveDetectorConfig = AdaptiveDetectorConfig(),
        direction_angles_deg: Optional[Mapping[int, float]] = None,
    ) -> None:
        self.config = config
        if direction_angles_deg is None:
            direction_angles_deg = {0: 270.0, 1: 90.0, 2: 180.0, 3: 0.0}
        self._direction_vectors = {
            int(code): (
                math.cos(math.radians(float(angle))),
                math.sin(math.radians(float(angle))),
            )
            for code, angle in direction_angles_deg.items()
        }
        lookup_size = max(4, max(self._direction_vectors, default=-1) + 1)
        self._direction_lookup = np.zeros((lookup_size, 2), dtype=float)
        self._direction_known = np.zeros(lookup_size, dtype=bool)
        for code, vector in self._direction_vectors.items():
            if code >= 0:
                self._direction_lookup[code] = vector
                self._direction_known[code] = True
        window_events = max(3, config.window_events)
        self._events: Deque[FlowEvent] = deque(maxlen=window_events)
        decay_age = np.arange(window_events - 1, -1, -1, dtype=float)
        if config.decay_events > 0.0:
            self._decay_weights = np.exp(-decay_age / config.decay_events)
        else:
            self._decay_weights = np.ones(window_events, dtype=float)
        self._event_index = -1
        self._last_update_index = -10**9
        self._last_detection: Optional[AdaptiveCircleDetection] = None
        self._track_detection: Optional[AdaptiveCircleDetection] = None
        self._pending_detection: Optional[_ScoredCircle] = None
        self._pending_count = 0
        self._track_misses = 0

    def reset(self) -> None:
        self._events.clear()
        self._event_index = -1
        self._last_update_index = -10**9
        self._last_detection = None
        self._track_detection = None
        self._pending_detection = None
        self._pending_count = 0
        self._track_misses = 0

    def push(self, event: FlowEvent) -> None:
        self._events.append(event)
        self._event_index += 1

    def extend(self, events: Iterable[FlowEvent]) -> None:
        for event in events:
            self.push(event)

    @property
    def last_detection(self) -> Optional[AdaptiveCircleDetection]:
        return self._last_detection

    @staticmethod
    def _circle_from_triples(
        x: np.ndarray,
        y: np.ndarray,
        triples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x1, x2, x3 = x[triples[:, 0]], x[triples[:, 1]], x[triples[:, 2]]
        y1, y2, y3 = y[triples[:, 0]], y[triples[:, 1]], y[triples[:, 2]]
        determinant = 2.0 * (
            x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        )
        safe_determinant = np.where(np.abs(determinant) > 1e-9, determinant, 1.0)
        squared1 = x1 * x1 + y1 * y1
        squared2 = x2 * x2 + y2 * y2
        squared3 = x3 * x3 + y3 * y3
        cx = (
            squared1 * (y2 - y3)
            + squared2 * (y3 - y1)
            + squared3 * (y1 - y2)
        ) / safe_determinant
        cy = (
            squared1 * (x3 - x2)
            + squared2 * (x1 - x3)
            + squared3 * (x2 - x1)
        ) / safe_determinant
        radius = np.hypot(x1 - cx, y1 - cy)

        edge12 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        edge23 = (x2 - x3) ** 2 + (y2 - y3) ** 2
        edge31 = (x3 - x1) ** 2 + (y3 - y1) ** 2
        largest_edge = np.maximum(np.maximum(edge12, edge23), edge31)
        triangle_quality = np.abs(determinant) / np.maximum(2.0 * largest_edge, 1e-9)
        valid = (np.abs(determinant) > 1e-6) & (triangle_quality >= 0.012)
        return cx, cy, radius, valid

    def _refine(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> tuple[float, float, float]:
        tolerance = self.config.radial_tolerance_px
        for _ in range(max(0, self.config.refine_iterations)):
            distance = np.hypot(x - cx, y - cy)
            residual = distance - radius
            mask = np.abs(residual) <= tolerance * 1.8
            if int(np.count_nonzero(mask)) < 10:
                break
            selected_distance = np.maximum(distance[mask], 1e-6)
            selected_residual = residual[mask]
            robust = np.minimum(
                1.0,
                tolerance / np.maximum(np.abs(selected_residual), 1e-6),
            )
            fit_weights = weights[mask] * robust
            jacobian = np.column_stack(
                (
                    (cx - x[mask]) / selected_distance,
                    (cy - y[mask]) / selected_distance,
                    -np.ones(int(np.count_nonzero(mask))),
                )
            )
            hessian = (jacobian * fit_weights[:, None]).T @ jacobian
            gradient = (jacobian * fit_weights[:, None]).T @ selected_residual
            try:
                step = np.linalg.solve(hessian + np.eye(3) * 1e-6, -gradient)
            except np.linalg.LinAlgError:
                break
            centre_step = math.hypot(float(step[0]), float(step[1]))
            if centre_step > 2.0:
                step[:2] *= 2.0 / centre_step
            cx += float(step[0])
            cy += float(step[1])
            radius += float(step[2])
            if float(np.linalg.norm(step)) < 0.03:
                break
        return cx, cy, radius

    def _refine_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        cx: np.ndarray,
        cy: np.ndarray,
        radius: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine several candidates with one set of NumPy operations."""

        cx = np.asarray(cx, dtype=float).copy()
        cy = np.asarray(cy, dtype=float).copy()
        radius = np.asarray(radius, dtype=float).copy()
        active = np.ones(len(radius), dtype=bool)
        tolerance = self.config.radial_tolerance_px
        identity = np.eye(3, dtype=float) * 1e-6

        for _ in range(max(0, self.config.refine_iterations)):
            distance = np.hypot(x[None, :] - cx[:, None], y[None, :] - cy[:, None])
            residual = distance - radius[:, None]
            mask = (np.abs(residual) <= tolerance * 1.8) & active[:, None]
            usable = active & (np.count_nonzero(mask, axis=1) >= 10)
            if not np.any(usable):
                break

            safe_distance = np.maximum(distance, 1e-6)
            robust = np.minimum(
                1.0,
                tolerance / np.maximum(np.abs(residual), 1e-6),
            )
            fit_weights = weights[None, :] * robust * mask
            jx = (cx[:, None] - x[None, :]) / safe_distance
            jy = (cy[:, None] - y[None, :]) / safe_distance

            hessian = np.empty((len(radius), 3, 3), dtype=float)
            hessian[:, 0, 0] = np.sum(fit_weights * jx * jx, axis=1)
            hessian[:, 0, 1] = hessian[:, 1, 0] = np.sum(
                fit_weights * jx * jy, axis=1
            )
            hessian[:, 0, 2] = hessian[:, 2, 0] = -np.sum(
                fit_weights * jx, axis=1
            )
            hessian[:, 1, 1] = np.sum(fit_weights * jy * jy, axis=1)
            hessian[:, 1, 2] = hessian[:, 2, 1] = -np.sum(
                fit_weights * jy, axis=1
            )
            hessian[:, 2, 2] = np.sum(fit_weights, axis=1)
            gradient = np.column_stack(
                (
                    np.sum(fit_weights * jx * residual, axis=1),
                    np.sum(fit_weights * jy * residual, axis=1),
                    -np.sum(fit_weights * residual, axis=1),
                )
            )

            step = np.zeros((len(radius), 3), dtype=float)
            selected = np.flatnonzero(usable)
            try:
                step[selected] = np.linalg.solve(
                    hessian[selected] + identity,
                    -gradient[selected, :, None],
                )[:, :, 0]
            except np.linalg.LinAlgError:
                for candidate_index in selected:
                    try:
                        step[candidate_index] = np.linalg.solve(
                            hessian[candidate_index] + identity,
                            -gradient[candidate_index],
                        )
                    except np.linalg.LinAlgError:
                        usable[candidate_index] = False

            centre_step = np.hypot(step[:, 0], step[:, 1])
            scale = np.minimum(1.0, 2.0 / np.maximum(centre_step, 1e-12))
            step[:, :2] *= scale[:, None]
            cx[usable] += step[usable, 0]
            cy[usable] += step[usable, 1]
            radius[usable] += step[usable, 2]
            converged = np.linalg.norm(step, axis=1) < 0.03
            active &= usable & ~converged

        return cx, cy, radius

    def _direction_agreement(
        self,
        x: np.ndarray,
        y: np.ndarray,
        codes: np.ndarray,
        mask: np.ndarray,
        cx: float,
        cy: float,
        weights: np.ndarray,
    ) -> float:
        selected = np.flatnonzero(mask)
        if not len(selected):
            return 0.0
        dx = x[selected] - cx
        dy = y[selected] - cy
        distance = np.maximum(np.hypot(dx, dy), 1e-6)
        agreement = np.zeros(len(selected), dtype=float)
        selected_codes = codes[selected].astype(np.int64, copy=False)
        known = (
            (selected_codes >= 0)
            & (selected_codes < len(self._direction_lookup))
        )
        known_indices = np.flatnonzero(known)
        if len(known_indices):
            known_codes = selected_codes[known_indices]
            known_indices = known_indices[self._direction_known[known_codes]]
            known_codes = selected_codes[known_indices]
            vectors = self._direction_lookup[known_codes]
            alignment = np.abs(
                (
                    vectors[:, 0] * dx[known_indices]
                    + vectors[:, 1] * dy[known_indices]
                )
                / distance[known_indices]
            )
            agreement[known_indices] = alignment >= math.sqrt(0.5)
        selected_weights = weights[selected]
        total = float(np.sum(selected_weights))
        return float(np.dot(agreement, selected_weights) / total) if total > 0.0 else 0.0

    def _score_circle(
        self,
        x: np.ndarray,
        y: np.ndarray,
        codes: np.ndarray,
        weights: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        *,
        continuation: bool = False,
    ) -> Optional[_ScoredCircle]:
        config = self.config
        if not (
            config.min_radius_px <= radius <= config.max_radius_px
            and -config.centre_margin_px <= cx <= config.width - 1 + config.centre_margin_px
            and -config.centre_margin_px <= cy <= config.height - 1 + config.centre_margin_px
        ):
            return None
        distance = np.hypot(x - cx, y - cy)
        deviation = np.abs(distance - radius)
        mask = deviation <= config.radial_tolerance_px
        inlier_count = int(np.count_nonzero(mask))
        inlier_scale = config.continuation_inlier_scale if continuation else 1.0
        ratio_scale = config.continuation_ratio_scale if continuation else 1.0
        sector_relaxation = (
            max(0, config.continuation_sector_relaxation) if continuation else 0
        )
        confidence_scale = (
            config.continuation_confidence_scale if continuation else 1.0
        )
        full_min_inliers = max(3, math.ceil(config.min_inliers * inlier_scale))
        upper_min_inliers = max(
            3, math.ceil(config.upper_min_inliers * inlier_scale)
        )
        if inlier_count < min(full_min_inliers, upper_min_inliers):
            return None
        total_weight = float(np.sum(weights))
        inlier_weight = float(np.sum(weights[mask]))
        ratio = inlier_weight / total_weight if total_weight > 0.0 else 0.0

        inlier_dx = x[mask] - cx
        inlier_dy = y[mask] - cy
        angle = np.arctan2(inlier_dy, inlier_dx)
        sector_ids = (
            (
                (angle + math.pi)
                / (2.0 * math.pi)
                * max(1, config.angular_sector_count)
            ).astype(int)
            % max(1, config.angular_sector_count)
        )
        sector_weights = np.bincount(
            sector_ids,
            weights=weights[mask],
            minlength=max(1, config.angular_sector_count),
        )
        quadrants = np.unique(
            ((angle + math.pi) / (2.0 * math.pi) * 4).astype(int) % 4
        )
        # One stray event must not claim an angular sector. This rejects the
        # common false large circle made from an image-border L shape plus a
        # handful of accidental inliers.
        sector_count = int(
            np.count_nonzero(sector_weights >= config.min_angular_sector_weight)
        )
        quadrant_count = int(len(quadrants))

        full_support = (
            inlier_count >= full_min_inliers
            and ratio >= config.min_inlier_weight_ratio * ratio_scale
            and sector_count
            >= max(3, config.min_angular_sectors - sector_relaxation)
            and quadrant_count >= config.min_quadrants
        )

        # A cue-mounted camera hides the lower half of a ball. Validate an
        # upper arc by requiring left shoulder, apex and right shoulder support;
        # this is much safer than merely lowering the total sector count.
        # Do not admit points just below the candidate centre here. Their
        # positive angles wrap near +pi and, if clipped, can fabricate a
        # 180-degree upper-arc span from an image-border L shape.
        upper_mask = inlier_dy <= 0.0
        upper_angles = np.clip(angle[upper_mask], -math.pi, 0.0)
        if len(upper_angles):
            upper_sector_ids = np.clip(
                (
                    (upper_angles + math.pi)
                    / math.pi
                    * max(1, config.angular_sector_count // 2)
                ).astype(int),
                0,
                max(0, config.angular_sector_count // 2 - 1),
            )
            upper_sector_weights = np.bincount(
                upper_sector_ids,
                weights=weights[mask][upper_mask],
                minlength=max(1, config.angular_sector_count // 2),
            )
            upper_sector_count = int(
                np.count_nonzero(
                    upper_sector_weights >= config.min_angular_sector_weight
                )
            )
            upper_span_deg = math.degrees(
                float(np.max(upper_angles) - np.min(upper_angles))
            )
        else:
            upper_sector_count = 0
            upper_span_deg = 0.0
        has_left_shoulder = bool(
            np.any(inlier_dx[upper_mask] <= -config.upper_endpoint_radius_ratio * radius)
        )
        has_right_shoulder = bool(
            np.any(inlier_dx[upper_mask] >= config.upper_endpoint_radius_ratio * radius)
        )
        has_apex = bool(
            np.any(inlier_dy[upper_mask] <= -config.upper_apex_radius_ratio * radius)
        )
        upper_support = (
            config.allow_upper_arc
            and inlier_count >= upper_min_inliers
            and ratio >= config.upper_min_inlier_weight_ratio * ratio_scale
            and upper_sector_count
            >= max(3, config.upper_min_angular_sectors - sector_relaxation)
            and upper_span_deg >= config.upper_min_span_deg
            and has_left_shoulder
            and has_right_shoulder
            and has_apex
        )
        if not full_support and not upper_support:
            return None

        radial_mad = float(np.median(deviation[mask]))
        if radial_mad > config.max_radial_mad_px:
            return None
        compactness = math.exp(-radial_mad / max(config.radial_tolerance_px, 1e-6))
        direction_agreement = self._direction_agreement(
            x, y, codes, mask, cx, cy, weights
        )
        if direction_agreement < config.min_direction_agreement:
            return None
        # Direction contributes only a small, centred term.  A random 4-way
        # flow field therefore neither validates nor destroys a geometric fit.
        direction_factor = 0.95 + 0.10 * max(0.0, min(1.0, direction_agreement))
        if full_support:
            support_mode = "full"
            coverage = min(
                1.0,
                sector_count / max(1.0, config.angular_sector_count / 2.0),
            )
            quadrant_coverage = min(
                1.0, quadrant_count / max(1.0, config.min_quadrants)
            )
            confidence = (
                ratio
                * coverage
                * quadrant_coverage
                * compactness
                * direction_factor
            )
            minimum_confidence = config.min_confidence * confidence_scale
        else:
            support_mode = "upper"
            span_fraction = np.clip(
                (upper_span_deg - config.upper_min_span_deg)
                / max(1.0, 180.0 - config.upper_min_span_deg),
                0.0,
                1.0,
            )
            coverage = 0.78 + 0.22 * float(span_fraction)
            confidence = (
                ratio
                * coverage
                * compactness
                * direction_factor
                * config.upper_confidence_scale
            )
            minimum_confidence = config.upper_min_confidence * confidence_scale
        if confidence < minimum_confidence:
            return None
        return _ScoredCircle(
            cx=float(cx),
            cy=float(cy),
            radius=float(radius),
            confidence=float(min(1.0, confidence)),
            inlier_count=inlier_count,
            radial_ratio=float(ratio),
            radial_mad=radial_mad,
            angular_sectors=sector_count,
            quadrants=quadrant_count,
            upper_angular_sectors=upper_sector_count,
            support_mode=support_mode,
            direction_agreement=direction_agreement,
        )

    def _is_continuous(
        self,
        candidate: _ScoredCircle,
        previous: AdaptiveCircleDetection,
    ) -> bool:
        config = self.config
        centre_limit = max(
            config.max_center_jump_px,
            previous.radius * config.max_center_jump_radius_ratio,
        )
        radius_limit = max(
            config.max_radius_jump_px,
            previous.radius * config.max_radius_jump_ratio,
        )
        return (
            math.hypot(candidate.cx - previous.cx, candidate.cy - previous.cy)
            <= centre_limit
            and abs(candidate.radius - previous.radius) <= radius_limit
        )

    def _update_pending(self, candidate: _ScoredCircle) -> None:
        pending = self._pending_detection
        if pending is not None and (
            math.hypot(candidate.cx - pending.cx, candidate.cy - pending.cy)
            <= self.config.pending_center_tolerance_px
            and abs(candidate.radius - pending.radius)
            <= self.config.pending_radius_tolerance_px
        ):
            self._pending_count += 1
            # Keep the strongest representative of the consistent pending track.
            if candidate.confidence >= pending.confidence:
                self._pending_detection = candidate
        else:
            self._pending_detection = candidate
            self._pending_count = 1

    def _make_detection(
        self,
        scored: _ScoredCircle,
        event_count: int,
        hypotheses_tested: int,
        timestamp: float,
    ) -> AdaptiveCircleDetection:
        return AdaptiveCircleDetection(
            cx=scored.cx,
            cy=scored.cy,
            radius=scored.radius,
            confidence=scored.confidence,
            event_count=event_count,
            inlier_count=scored.inlier_count,
            radial_inlier_ratio=scored.radial_ratio,
            radial_mad=scored.radial_mad,
            angular_sectors=scored.angular_sectors,
            quadrants=scored.quadrants,
            upper_angular_sectors=scored.upper_angular_sectors,
            support_mode=scored.support_mode,
            direction_agreement=scored.direction_agreement,
            hypotheses_tested=hypotheses_tested,
            timestamp=timestamp,
            event_index=self._event_index,
        )

    def _accept_track(
        self,
        scored: _ScoredCircle,
        event_count: int,
        hypotheses_tested: int,
        timestamp: float,
    ) -> AdaptiveCircleDetection:
        detection = self._make_detection(
            scored, event_count, hypotheses_tested, timestamp
        )
        self._track_detection = detection
        self._track_misses = 0
        self._pending_detection = None
        self._pending_count = 0
        return detection

    def _handle_missing_track(
        self,
        x: np.ndarray,
        y: np.ndarray,
        codes: np.ndarray,
        weights: np.ndarray,
        event_count: int,
        hypotheses_tested: int,
        timestamp: float,
    ) -> Optional[AdaptiveCircleDetection]:
        previous = self._track_detection
        if previous is None:
            return None
        held = self._score_circle(
            x,
            y,
            codes,
            weights,
            previous.cx,
            previous.cy,
            previous.radius,
        )
        if held is not None:
            return self._accept_track(
                held, event_count, hypotheses_tested, timestamp
            )
        continued = self._score_circle(
            x,
            y,
            codes,
            weights,
            previous.cx,
            previous.cy,
            previous.radius,
            continuation=True,
        )
        self._track_misses += 1
        reset_updates = max(1, self.config.track_reset_updates)
        if self._track_misses >= reset_updates:
            self._track_detection = None
            self._pending_detection = None
            self._pending_count = 0
        # Leave at least one blank update before a possible confirmed track
        # switch, otherwise a filled dropout could visually reconnect two
        # different circles and reintroduce a one-event jump.
        if continued is None or self._track_misses >= max(1, reset_updates - 1):
            return None
        # This relaxed result fills a short visual dropout only. Deliberately
        # do not update track state: strict tracking still accumulates the same
        # miss count and resets exactly as it would without continuation mode.
        return self._make_detection(
            continued, event_count, hypotheses_tested, timestamp
        )

    def _estimate(self) -> Optional[AdaptiveCircleDetection]:
        config = self.config
        samples = list(self._events)
        if len(samples) < config.min_events:
            return None
        x = np.asarray([event.x for event in samples], dtype=float)
        y = np.asarray([event.y for event in samples], dtype=float)
        codes = np.asarray([event.c for event in samples], dtype=np.int16)
        weights = self._decay_weights[-len(samples) :]

        rng = np.random.default_rng(
            int(config.seed + 1_000_003 * max(0, self._event_index))
        )
        triples = rng.integers(0, len(samples), size=(max(1, config.hypotheses), 3))
        cx, cy, radius, valid = self._circle_from_triples(x, y, triples)
        margin = config.centre_margin_px
        valid &= radius >= config.min_radius_px
        valid &= radius <= config.max_radius_px
        valid &= cx >= -margin
        valid &= cx <= config.width - 1 + margin
        valid &= cy >= -margin
        valid &= cy <= config.height - 1 + margin
        cx = cx[valid]
        cy = cy[valid]
        radius = radius[valid]
        if not len(radius):
            return None

        if self._track_detection is not None:
            cx = np.append(cx, self._track_detection.cx)
            cy = np.append(cy, self._track_detection.cy)
            radius = np.append(radius, self._track_detection.radius)

        residual = np.abs(
            np.hypot(x[None, :] - cx[:, None], y[None, :] - cy[:, None])
            - radius[:, None]
        )
        support = (residual <= config.radial_tolerance_px) @ weights
        count = min(max(1, config.refine_candidates), len(support))
        top_indices = np.argpartition(support, -count)[-count:]
        if self._track_detection is not None:
            top_indices = np.unique(np.append(top_indices, len(support) - 1))

        best: Optional[_ScoredCircle] = None
        best_rank = -math.inf
        previous = self._track_detection
        refined_cx, refined_cy, refined_radius = self._refine_batch(
            x,
            y,
            weights,
            cx[top_indices],
            cy[top_indices],
            radius[top_indices],
        )
        for refined in zip(refined_cx, refined_cy, refined_radius):
            scored = self._score_circle(x, y, codes, weights, *refined)
            if scored is None:
                continue
            rank = scored.confidence
            if previous is not None:
                centre_delta = math.hypot(scored.cx - previous.cx, scored.cy - previous.cy)
                radius_delta = abs(scored.radius - previous.radius)
                centre_scale = max(6.0, previous.radius * 0.25)
                radius_scale = max(4.0, previous.radius * 0.18)
                rank += config.tracking_bonus * math.exp(
                    -centre_delta / centre_scale - radius_delta / radius_scale
                )
            if rank > best_rank:
                best_rank = rank
                best = scored
        if best is None:
            newest = samples[-1]
            return self._handle_missing_track(
                x,
                y,
                codes,
                weights,
                len(samples),
                int(len(radius)),
                float(newest.t),
            )

        if previous is not None:
            if self._is_continuous(best, previous):
                alpha = max(0.0, min(1.0, config.smoothing))
                smoothed = (
                    previous.cx * (1.0 - alpha) + best.cx * alpha,
                    previous.cy * (1.0 - alpha) + best.cy * alpha,
                    previous.radius * (1.0 - alpha) + best.radius * alpha,
                )
                rescored = self._score_circle(x, y, codes, weights, *smoothed)
                if rescored is not None:
                    best = rescored
            else:
                # A far candidate is never allowed to replace a supported track
                # in one update. It must remain consistent while the old track
                # is absent for several consecutive detector updates.
                self._update_pending(best)
                held = self._score_circle(
                    x,
                    y,
                    codes,
                    weights,
                    previous.cx,
                    previous.cy,
                    previous.radius,
                )
                if held is not None:
                    return self._accept_track(
                        held,
                        len(samples),
                        int(len(radius)),
                        float(samples[-1].t),
                    )
                self._track_misses += 1
                if (
                    self._track_misses >= max(1, config.track_reset_updates)
                    and self._pending_count >= max(1, config.jump_confirmations)
                    and self._pending_detection is not None
                ):
                    best = self._pending_detection
                    self._track_detection = None
                else:
                    return None

        newest = samples[-1]
        return self._accept_track(
            best,
            len(samples),
            int(len(radius)),
            float(newest.t),
        )

    def detect(self, force: bool = False) -> Optional[AdaptiveCircleDetection]:
        interval = max(1, self.config.update_interval_events)
        if not force and self._event_index - self._last_update_index < interval:
            return self._last_detection
        self._last_update_index = self._event_index
        self._last_detection = self._estimate()
        if self._last_detection is not None:
            detection = self._last_detection
            config = self.config
            # Apply the FULL expression even when the tracked geometry was
            # admitted through upper support. Do not inherit the upper bonus.
            full = (
                detection.radial_inlier_ratio
                * min(1.0, detection.angular_sectors / max(1, config.angular_sector_count/2))
                * min(1.0, detection.quadrants / max(1, config.min_quadrants))
                * math.exp(-detection.radial_mad / max(config.radial_tolerance_px, 1e-6))
                * (0.95 + 0.10*detection.direction_agreement)
            )
            samples = list(self._events)
            diagnostic = calculate_xiaoiron_confidence(
                [e.x for e in samples], [e.y for e in samples],
                [e.c for e in samples], self._decay_weights[-len(samples):],
                cx=detection.cx, cy=detection.cy, radius=detection.radius,
                radial_tolerance_px=config.radial_tolerance_px,
                full_confidence=full,
                config=replace(config.xiaoiron, width=config.width, height=config.height),
            )
            self._last_detection = replace(detection, xiaoiron=diagnostic)
        return self._last_detection
