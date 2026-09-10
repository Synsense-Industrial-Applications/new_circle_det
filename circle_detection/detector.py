"""Direction-separable circle detection for asynchronous flow events.

The detector does not scan circle radii.  A direction event constrains the
circle centre to the normal line through that event.  Opposite signed flow
directions are treated as the same unoriented normal family.  Robust 1-D
projection medians provide the line offsets, a 2x2 solve gives the centre,
and a final 1-D distance consensus gives the radius.

The reference implementation deliberately uses only the Python standard
library.  The same arithmetic maps directly to a small embedded C pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FlowEvent:
    """One DVS/SNN direction event.

    ``t`` and all temporal configuration values use the same unit.  For the
    repository CSV files that unit is normally microseconds.
    """

    x: float
    y: float
    c: int
    t: float


@dataclass(frozen=True)
class DetectorConfig:
    width: int = 128
    height: int = 128
    time_window: float = 20_000.0
    time_tau: float = 8_000.0
    max_events: int = 1024
    min_events: int = 28
    min_events_per_family: int = 6
    min_direction_families: int = 2
    direction_half_width_deg: float = 45.0
    direction_margin_deg: float = 8.0
    radial_tolerance_px: float = 3.0
    min_radial_inlier_ratio: float = 0.55
    max_radial_mad_px: float = 3.0
    min_angular_sectors: int = 4
    angular_sector_count: int = 12
    centre_margin_px: float = 2.0
    refine_iterations: int = 2
    max_refine_step_px: float = 2.0
    min_observability: float = 0.08


@dataclass(frozen=True)
class CircleDetection:
    cx: float
    cy: float
    radius: float
    confidence: float
    event_count: int
    inlier_count: int
    radial_inlier_ratio: float
    radial_mad: float
    direction_families: int
    angular_sectors: int
    observability: float
    timestamp: float


@dataclass(frozen=True)
class _DirectionInfo:
    raw_angle: float
    family_key: float
    nx: float
    ny: float
    qx: float
    qy: float


def _weighted_median(values: Sequence[Tuple[float, float]]) -> float:
    if not values:
        raise ValueError("weighted median needs at least one value")
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in ordered)
    if total <= 0.0:
        return ordered[len(ordered) // 2][0]
    halfway = total * 0.5
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= halfway:
            return value
    return ordered[-1][0]


def _weighted_mad(values: Sequence[Tuple[float, float]], centre: float) -> float:
    return _weighted_median([(abs(value - centre), weight) for value, weight in values])


def _canonical_angle(angle: float) -> float:
    value = math.fmod(angle, math.pi)
    if value < 0.0:
        value += math.pi
    if value >= math.pi - 1e-10:
        value = 0.0
    if abs(value) < 1e-10:
        value = 0.0
    return value


class DirectionalCircleDetector:
    """Streaming detector for one dominant circle at an arbitrary scale.

    ``direction_angles_deg`` maps each SNN direction code to the normal-flow
    angle in image coordinates.  The default mapping is right/down/left/up.
    Direction sign is intentionally ignored for centre geometry because DVS
    normal flow may point either towards or away from the circle centre.
    """

    def __init__(
        self,
        config: DetectorConfig = DetectorConfig(),
        direction_angles_deg: Optional[Mapping[int, float]] = None,
    ) -> None:
        if direction_angles_deg is None:
            direction_angles_deg = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}
        if len(direction_angles_deg) < 2:
            raise ValueError("at least two direction codes are required")
        self.config = config
        self._directions = self._make_direction_table(direction_angles_deg)
        self._events: Deque[FlowEvent] = deque(maxlen=config.max_events)

    @staticmethod
    def _make_direction_table(
        direction_angles_deg: Mapping[int, float],
    ) -> Dict[int, _DirectionInfo]:
        table: Dict[int, _DirectionInfo] = {}
        for code, angle_deg in direction_angles_deg.items():
            raw = math.radians(float(angle_deg))
            family_angle = _canonical_angle(raw)
            # Rounding makes exact antipodes share a stable dictionary key.
            family_key = round(family_angle, 9)
            nx = math.cos(family_angle)
            ny = math.sin(family_angle)
            table[int(code)] = _DirectionInfo(
                raw_angle=raw,
                family_key=family_key,
                nx=nx,
                ny=ny,
                qx=-ny,
                qy=nx,
            )
        return table

    def reset(self) -> None:
        self._events.clear()

    def push(self, event: FlowEvent) -> None:
        if event.c not in self._directions:
            return
        self._events.append(event)
        self._prune(event.t)

    def extend(self, events: Iterable[FlowEvent]) -> None:
        for event in events:
            self.push(event)

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.time_window
        while self._events and self._events[0].t < cutoff:
            self._events.popleft()

    def _snapshot(self, now: float) -> Sequence[Tuple[FlowEvent, float, _DirectionInfo]]:
        self._prune(now)
        tau = self.config.time_tau
        out = []
        for event in self._events:
            age = max(0.0, now - event.t)
            if age > self.config.time_window:
                continue
            weight = math.exp(-age / tau) if tau > 0.0 else 1.0
            out.append((event, weight, self._directions[event.c]))
        return out

    def _initial_centre(
        self,
        samples: Sequence[Tuple[FlowEvent, float, _DirectionInfo]],
    ) -> Optional[Tuple[float, float, int, float]]:
        grouped: Dict[float, list[Tuple[float, float]]] = {}
        for event, weight, info in samples:
            rho = info.qx * event.x + info.qy * event.y
            grouped.setdefault(info.family_key, []).append((rho, weight))

        constraints = []
        for family_key, values in grouped.items():
            if len(values) < self.config.min_events_per_family:
                continue
            rho = _weighted_median(values)
            mad = _weighted_mad(values, rho)
            total_weight = sum(weight for _, weight in values)
            # The median remains useful for broad quantisation sectors; cap
            # the spread penalty so large balls are not penalised excessively.
            spread = min(max(mad, 1.0), 8.0)
            constraint_weight = total_weight / (spread * spread)
            any_info = next(
                info
                for info in self._directions.values()
                if info.family_key == family_key
            )
            constraints.append((any_info.qx, any_info.qy, rho, constraint_weight))

        if len(constraints) < self.config.min_direction_families:
            return None

        a00 = a01 = a11 = b0 = b1 = 0.0
        for qx, qy, rho, weight in constraints:
            a00 += weight * qx * qx
            a01 += weight * qx * qy
            a11 += weight * qy * qy
            b0 += weight * qx * rho
            b1 += weight * qy * rho

        det = a00 * a11 - a01 * a01
        trace = a00 + a11
        if det <= 1e-12 or trace <= 1e-12:
            return None
        observability = max(0.0, min(1.0, 4.0 * det / (trace * trace)))
        if observability < self.config.min_observability:
            return None

        cx = (b0 * a11 - b1 * a01) / det
        cy = (a00 * b1 - a01 * b0) / det
        return cx, cy, len(constraints), observability

    def _direction_filtered(
        self,
        samples: Sequence[Tuple[FlowEvent, float, _DirectionInfo]],
        cx: float,
        cy: float,
    ) -> list[Tuple[FlowEvent, float, _DirectionInfo, float]]:
        max_angle = math.radians(
            self.config.direction_half_width_deg + self.config.direction_margin_deg
        )
        out = []
        for event, weight, info in samples:
            dx = event.x - cx
            dy = event.y - cy
            distance = math.hypot(dx, dy)
            if distance <= 1e-9:
                continue
            alignment = abs((info.nx * dx + info.ny * dy) / distance)
            angle_error = math.acos(max(0.0, min(1.0, alignment)))
            if angle_error <= max_angle:
                out.append((event, weight, info, distance))
        return out

    def _refine_centre_radius(
        self,
        samples: Sequence[Tuple[FlowEvent, float, _DirectionInfo]],
        cx: float,
        cy: float,
    ) -> Optional[Tuple[float, float, float, list[Tuple[FlowEvent, float, float]]]]:
        oriented = self._direction_filtered(samples, cx, cy)
        if len(oriented) < self.config.min_events:
            return None
        radius = _weighted_median([(distance, weight) for _, weight, _, distance in oriented])

        for _ in range(max(0, self.config.refine_iterations)):
            residual_values = [
                (abs(distance - radius), weight)
                for _, weight, _, distance in oriented
            ]
            mad = _weighted_median(residual_values)
            huber = max(self.config.radial_tolerance_px, 1.4826 * mad)
            h00 = h01 = h11 = g0 = g1 = 0.0
            for event, base_weight, _, distance in oriented:
                if distance <= 1e-9:
                    continue
                residual = distance - radius
                robust = 1.0 if abs(residual) <= huber else huber / abs(residual)
                weight = base_weight * robust
                jx = (cx - event.x) / distance
                jy = (cy - event.y) / distance
                h00 += weight * jx * jx
                h01 += weight * jx * jy
                h11 += weight * jy * jy
                g0 += weight * jx * residual
                g1 += weight * jy * residual
            det = h00 * h11 - h01 * h01
            if det <= 1e-9:
                break
            step_x = -(h11 * g0 - h01 * g1) / det
            step_y = -(-h01 * g0 + h00 * g1) / det
            step_norm = math.hypot(step_x, step_y)
            if step_norm > self.config.max_refine_step_px:
                scale = self.config.max_refine_step_px / step_norm
                step_x *= scale
                step_y *= scale
            cx += step_x
            cy += step_y
            oriented = self._direction_filtered(samples, cx, cy)
            if len(oriented) < self.config.min_events:
                return None
            radius = _weighted_median(
                [(distance, weight) for _, weight, _, distance in oriented]
            )
            if math.hypot(step_x, step_y) < 0.05:
                break

        compact = [(event, weight, distance) for event, weight, _, distance in oriented]
        return cx, cy, radius, compact

    def detect(self, now: Optional[float] = None) -> Optional[CircleDetection]:
        if not self._events:
            return None
        if now is None:
            now = self._events[-1].t
        samples = self._snapshot(float(now))
        if len(samples) < self.config.min_events:
            return None

        initial = self._initial_centre(samples)
        if initial is None:
            return None
        cx, cy, family_count, observability = initial
        margin = self.config.centre_margin_px
        if not (-margin <= cx <= self.config.width - 1 + margin):
            return None
        if not (-margin <= cy <= self.config.height - 1 + margin):
            return None

        refined = self._refine_centre_radius(samples, cx, cy)
        if refined is None:
            return None
        cx, cy, radius, oriented = refined
        if radius <= 0.0:
            return None

        tolerance = self.config.radial_tolerance_px
        inliers = [
            (event, weight, distance)
            for event, weight, distance in oriented
            if abs(distance - radius) <= tolerance
        ]
        if len(inliers) < self.config.min_events:
            return None

        total_weight = sum(weight for _, weight, _ in samples)
        inlier_weight = sum(weight for _, weight, _ in inliers)
        radial_ratio = inlier_weight / total_weight if total_weight > 0.0 else 0.0
        radial_mad = _weighted_median(
            [(abs(distance - radius), weight) for _, weight, distance in oriented]
        )
        if radial_ratio < self.config.min_radial_inlier_ratio:
            return None
        if radial_mad > self.config.max_radial_mad_px:
            return None

        sector_count = max(1, self.config.angular_sector_count)
        sectors = set()
        for event, _, _ in inliers:
            angle = math.atan2(event.y - cy, event.x - cx)
            index = int(((angle + math.pi) / (2.0 * math.pi)) * sector_count)
            sectors.add(index % sector_count)
        angular_sectors = len(sectors)
        if angular_sectors < self.config.min_angular_sectors:
            return None

        oriented_weight = sum(weight for _, weight, _ in oriented)
        orientation_ratio = oriented_weight / total_weight if total_weight > 0.0 else 0.0
        coverage = min(1.0, angular_sectors / max(1.0, sector_count / 2.0))
        confidence = radial_ratio
        confidence *= math.sqrt(max(0.0, min(1.0, orientation_ratio)))
        confidence *= math.sqrt(observability)
        confidence *= coverage
        confidence = max(0.0, min(1.0, confidence))

        return CircleDetection(
            cx=cx,
            cy=cy,
            radius=radius,
            confidence=confidence,
            event_count=len(samples),
            inlier_count=len(inliers),
            radial_inlier_ratio=radial_ratio,
            radial_mad=radial_mad,
            direction_families=family_count,
            angular_sectors=angular_sectors,
            observability=observability,
            timestamp=float(now),
        )
