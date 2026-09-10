"""Inspectable cue-camera confidence, built on the full-circle geometric score.

Sector IDs are ZERO based, matching adaptive_detector and 圆弧.txt:
floor((atan2(y-cy, x-cx) + pi) / (2*pi) * 16) % 16.
Only events within the candidate's radial tolerance are counted.
This is a heuristic score, not a calibrated probability of a hit.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class XiaoironConfig:
    occluded_sectors: tuple[int, ...] = (10, 11, 12, 13)
    direction_sectors: tuple[int, ...] = (1, 2, 5, 6)
    occlusion_gain: float = 0.51
    direction_penalty_lambda: float = 0.85
    min_direction_events: int = 5
    min_direction_weight: float = 1.75
    min_other_sectors: int = 6
    min_sector_weight: float = 1.0
    width: int = 128
    height: int = 128

    def __post_init__(self):
        if not self.occluded_sectors or not self.direction_sectors:
            raise ValueError("sector groups must not be empty")
        if len(self.direction_sectors) != 4:
            raise ValueError("the direction rule requires exactly four sectors")
        for group in (self.occluded_sectors, self.direction_sectors):
            if len(set(group)) != len(group) or any(
                not isinstance(s, (int, np.integer)) or s not in range(16) for s in group
            ):
                raise ValueError("sector IDs must be unique integers in 0..15")
        parameters = (self.occlusion_gain, self.direction_penalty_lambda,
                      self.min_direction_events, self.min_direction_weight,
                      self.min_other_sectors, self.min_sector_weight,
                      self.width, self.height)
        if not all(math.isfinite(p) for p in parameters):
            raise ValueError("configuration values must be finite")
        if self.occlusion_gain < 0 or self.direction_penalty_lambda < 0:
            raise ValueError("gains must be nonnegative")
        if min(self.min_direction_events, self.min_direction_weight,
               self.min_other_sectors, self.min_sector_weight) <= 0:
            raise ValueError("evidence thresholds must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sensor size must be positive")


@dataclass(frozen=True)
class XiaoironScore:
    full_confidence: float
    xiaoiron_confidence: float
    occlusion_factor: float
    direction_factor: float
    evidence_strength: float
    direction_evidence: float
    direction_gap: float
    side_sufficiency: np.ndarray
    side_direction_evidence: np.ndarray
    empty_visible_count: int
    sector_counts: np.ndarray
    direction_weights: np.ndarray
    sector_purity: np.ndarray
    sector_visible: np.ndarray


def sector_ids(dx, dy):
    return (np.floor((np.arctan2(dy, dx) + np.pi) * (16 / (2*np.pi)))
            .astype(np.int64) % 16)


def calculate_xiaoiron_confidence(
    x, y, codes, weights, *, cx, cy, radius, radial_tolerance_px,
    full_confidence, config=XiaoironConfig(),
):
    """One pure function: event window + fixed circle + full score -> diagnostics.

    E = any of (10,11,12,13) has EXACTLY zero ring events and is inside the sensor.
    G = min(all four direction sectors' sufficiency, other-sector coverage).
    R = min(q_s*p_s for s in (1,2,5,6)), combining sufficiency and purity.
    C_x = clip(C_full * (1 + alpha*E*G) * exp(-lambda*(1-R)), 0, 1).
    Empty direction sectors are unknown (purity NaN) and contribute q*p = 0.
    """
    x, y, codes, weights = map(np.asarray, (x, y, codes, weights))
    if not (x.ndim == y.ndim == codes.ndim == weights.ndim == 1
            and len(x) == len(y) == len(codes) == len(weights)):
        raise ValueError("x/y/codes/weights must be equally sized one-dimensional arrays")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("event positions must be finite")
    if not all(math.isfinite(v) for v in (cx, cy, radius, radial_tolerance_px, full_confidence)):
        raise ValueError("circle parameters and full_confidence must be finite")
    if radius <= 0 or radial_tolerance_px <= 0:
        raise ValueError("radius and radial tolerance must be positive")
    if not np.all(np.isfinite(codes)) or np.any((codes < 0) | (codes > 3) | (codes != np.floor(codes))):
        raise ValueError("direction codes must be integers 0..3")
    mask = np.abs(np.hypot(x-cx, y-cy) - radius) <= radial_tolerance_px
    ids = sector_ids(x[mask]-cx, y[mask]-cy)
    counts = np.bincount(ids, minlength=16)
    hist = np.bincount(
        ids*4 + codes[mask].astype(int), weights=weights[mask], minlength=64
    ).reshape(16, 4)
    mass = hist.sum(axis=1)
    purity = np.divide(hist.max(axis=1), mass,
                       out=np.full(16, np.nan), where=mass > 0)

    # Exact arc extrema occur at endpoints or multiples of 90 degrees.
    # Require the whole tolerance band to remain inside the image: sensor
    # clipping must never be interpreted as evidence of cue occlusion.
    visible = np.zeros(16, dtype=bool)
    for s in range(16):
        a, b = -math.pi + s*math.pi/8, -math.pi + (s+1)*math.pi/8
        angles = [a, b] + [k*math.pi/2 for k in range(-2, 3)
                           if a <= k*math.pi/2 <= b]
        px = cx + radius*np.cos(angles)
        py = cy + radius*np.sin(angles)
        d = radial_tolerance_px
        visible[s] = (px.min() >= d and px.max() <= config.width-1-d
                      and py.min() >= d and py.max() <= config.height-1-d)
    occ = np.asarray(config.occluded_sectors, dtype=int)
    sides = np.asarray(config.direction_sectors, dtype=int)
    empty_count = int(np.count_nonzero((counts[occ] == 0) & visible[occ]))
    strength = np.minimum(
        1.0, np.minimum(counts[sides]/config.min_direction_events,
                        mass[sides]/config.min_direction_weight))
    other = np.ones(16, dtype=bool)
    other[occ] = False
    coverage = min(1.0, np.count_nonzero(
        mass[other] >= config.min_sector_weight) / config.min_other_sectors)
    evidence = float(min(float(strength.min()), coverage))
    side_purity = np.nan_to_num(purity[sides], nan=0.0)
    side_evidence = strength * side_purity
    direction_evidence = float(side_evidence.min())
    direction_gap = 1.0 - direction_evidence
    occlusion_factor = 1.0 + config.occlusion_gain*(empty_count > 0)*evidence
    direction_factor = math.exp(-config.direction_penalty_lambda*direction_gap)
    full = float(np.clip(full_confidence, 0, 1))
    return XiaoironScore(
        full, float(np.clip(full*occlusion_factor*direction_factor, 0, 1)),
        occlusion_factor, direction_factor, evidence,
        direction_evidence, direction_gap, strength, side_evidence, empty_count,
        counts, hist, purity, visible,
    )
