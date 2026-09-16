#!/usr/bin/env python3
"""Decode the 13-point four-region Layer-4 event representation.

This module deliberately mirrors ``event_stream_player.py``. The source
address is 64x64 and feature 0..15 contains two spatial-kernel banks.  Both
banks retain the original 2x2 sub-pixel address and four-direction flow code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_SIZE = 128
FEATURE_TO_DIRECTION = np.asarray(
    [0, 1, 1, 0, 2, 3, 3, 2] * 2,
    dtype=np.int8,
)
LAYER4_FEATURE_COUNT = len(FEATURE_TO_DIRECTION)
DIRECTION_NAMES = ("down-right", "down-left", "up-left", "up-right")
DIRECTION_SYMBOLS = ("↘", "↙", "↖", "↗")
DIRECTION_ANGLES_DEG = {0: 45.0, 1: 135.0, 2: 225.0, 3: 315.0}


@dataclass(frozen=True)
class FlowData:
    path: Path
    x: np.ndarray
    y: np.ndarray
    direction: np.ndarray
    timestamp_us: np.ndarray
    relative_s: np.ndarray
    feature: np.ndarray
    layer: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.relative_s[-1]) if len(self.relative_s) else 0.0

    @property
    def direction_codes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(self.direction))


def _field(fields: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _parse_int(value: str, column: str, row_number: int, path: Path) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name}: row {row_number} has invalid {column}={value!r}"
        ) from exc


def load_flow_csv(path: Path) -> FlowData:
    """Load CSV and return stably time-sorted 128x128 ``(x, y, c, t)`` data."""

    path = Path(path)
    raw_x: list[int] = []
    raw_y: list[int] = []
    features: list[int] = []
    timestamps: list[int] = []
    layers: list[int] = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {
            str(name).strip().lower(): name for name in (reader.fieldnames or ())
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
        layer_field = _field(fields, "layer")
        if timestamp_field is None or x_field is None or y_field is None:
            raise ValueError("CSV must contain timestamp, x and y columns")
        if feature_field is None:
            raise ValueError("13-point four-region CSV must contain feature")

        for row_number, row in enumerate(reader, start=2):
            raw_x.append(_parse_int(row[x_field], "x", row_number, path))
            raw_y.append(_parse_int(row[y_field], "y", row_number, path))
            features.append(
                _parse_int(row[feature_field], "feature", row_number, path)
            )
            timestamps.append(
                _parse_int(row[timestamp_field], "timestamp", row_number, path)
            )
            layers.append(
                _parse_int(row[layer_field], "layer", row_number, path)
                if layer_field is not None
                else 0
            )

    if not timestamps:
        raise ValueError(f"{path.name}: no events")

    x64 = np.asarray(raw_x, dtype=np.int16)
    y64 = np.asarray(raw_y, dtype=np.int16)
    feature = np.asarray(features, dtype=np.int16)
    timestamp_us = np.asarray(timestamps, dtype=np.int64)
    layer = np.asarray(layers, dtype=np.int16)

    if np.any((feature < 0) | (feature >= LAYER4_FEATURE_COUNT)):
        raise ValueError(
            f"{path.name}: feature must be in 0..{LAYER4_FEATURE_COUNT - 1}"
        )
    if np.any((x64 < 0) | (x64 >= 64) | (y64 < 0) | (y64 >= 64)):
        raise ValueError(f"{path.name}: expected 64x64 coordinates in 0..63")

    feature_mod4 = feature.astype(np.int32) % 4
    x128 = x64.astype(np.int32) * 2 + (feature_mod4 // 2) % 2
    y128 = y64.astype(np.int32) * 2 + feature_mod4 % 2
    direction = FEATURE_TO_DIRECTION[feature]

    order = np.argsort(timestamp_us, kind="stable")
    timestamp_us = timestamp_us[order]
    relative_s = (timestamp_us - timestamp_us[0]).astype(np.float64) * 1e-6
    return FlowData(
        path=path,
        x=x128[order].astype(np.int16, copy=False),
        y=y128[order].astype(np.int16, copy=False),
        direction=direction[order].astype(np.int8, copy=False),
        timestamp_us=timestamp_us,
        relative_s=relative_s,
        feature=feature[order],
        layer=layer[order],
    )


def export_decoded_csv(data: FlowData, output: Path | None = None) -> Path:
    """Export decoded ``x,y,c,t`` values for inspection or C-side replay."""

    output = Path(output) if output is not None else data.path.with_name(
        f"{data.path.stem}_decoded_flow.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "c", "t", "direction", "source_feature"))
        for x, y, code, timestamp, feature in zip(
            data.x,
            data.y,
            data.direction,
            data.timestamp_us,
            data.feature,
        ):
            writer.writerow(
                (
                    int(x),
                    int(y),
                    int(code),
                    int(timestamp),
                    DIRECTION_NAMES[int(code)],
                    int(feature),
                )
            )
    return output
