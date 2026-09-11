"""Temporary, low-overhead CSV logging for real-machine playback measurements.

Delete this file and the small ``performance_logger`` hooks in
``visualize_circle_detection.py`` after the measurements are collected.
"""

from __future__ import annotations

import atexit
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Optional


class RuntimePerformanceLogger:
    FIELDNAMES = (
        "session_id",
        "record_type",
        "wall_time",
        "event_number",
        "source_timestamp_us",
        "source_relative_s",
        "events_advanced",
        "source_advance_us",
        "target_lag_us",
        "visible_events",
        "wall_interval_ms",
        "actual_fps",
        "effective_source_speed",
        "callback_ms",
        "render_update_ms",
        "draw_latency_ms",
        "playback_speed",
        "timer_interval_ms",
        "score_window_events",
        "duration_ms",
        "detail",
    )

    def __init__(self, path: Path, *, flush_every: int = 25) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file = self.path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._flush_every = max(1, int(flush_every))
        self._rows_since_flush = 0
        self._last_tick_s: Optional[float] = None
        self._last_tick_source_us: Optional[float] = None
        self._last_draw_s: Optional[float] = None
        self._closed = False
        atexit.register(self.close)
        self.record(
            "session_start",
            detail=json.dumps(
                {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def record(self, record_type: str, **values: Any) -> None:
        if self._closed:
            return
        row = {name: "" for name in self.FIELDNAMES}
        row.update(values)
        row["session_id"] = self.session_id
        row["record_type"] = record_type
        row["wall_time"] = datetime.now().isoformat(timespec="milliseconds")
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self._file.flush()
            self._rows_since_flush = 0

    def record_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.record(
            "metadata",
            detail=json.dumps(
                dict(metadata), ensure_ascii=False, separators=(",", ":")
            ),
        )

    def record_stage(self, name: str, duration_ms: float, detail: str = "") -> None:
        self.record(
            "stage",
            duration_ms=f"{duration_ms:.3f}",
            detail=f"{name}: {detail}" if detail else name,
        )

    def record_tick(
        self,
        *,
        now_s: float,
        event_number: int,
        source_timestamp_us: float,
        source_relative_s: float,
        events_advanced: int,
        target_lag_us: float,
        visible_events: int,
        callback_ms: float,
        render_update_ms: float,
        playback_speed: float,
        timer_interval_ms: int,
        score_window_events: int,
    ) -> None:
        wall_interval_ms = ""
        actual_fps = ""
        source_advance_us = ""
        effective_source_speed = ""
        if self._last_tick_s is not None:
            wall_delta_s = max(0.0, now_s - self._last_tick_s)
            wall_interval_ms = f"{wall_delta_s * 1000.0:.3f}"
            if wall_delta_s > 0.0:
                actual_fps = f"{1.0 / wall_delta_s:.3f}"
            if self._last_tick_source_us is not None:
                source_delta_us = source_timestamp_us - self._last_tick_source_us
                source_advance_us = f"{source_delta_us:.3f}"
                if wall_delta_s > 0.0:
                    effective_source_speed = f"{source_delta_us / (wall_delta_s * 1e6):.5f}"
        self._last_tick_s = now_s
        self._last_tick_source_us = source_timestamp_us
        self.record(
            "timer_tick",
            event_number=event_number,
            source_timestamp_us=f"{source_timestamp_us:.3f}",
            source_relative_s=f"{source_relative_s:.6f}",
            events_advanced=events_advanced,
            source_advance_us=source_advance_us,
            target_lag_us=f"{target_lag_us:.3f}",
            visible_events=visible_events,
            wall_interval_ms=wall_interval_ms,
            actual_fps=actual_fps,
            effective_source_speed=effective_source_speed,
            callback_ms=f"{callback_ms:.3f}",
            render_update_ms=f"{render_update_ms:.3f}",
            playback_speed=f"{playback_speed:.5f}",
            timer_interval_ms=timer_interval_ms,
            score_window_events=score_window_events,
        )

    def record_draw(
        self,
        *,
        now_s: float,
        event_number: int,
        source_timestamp_us: float,
        pending_tick_started_s: Optional[float],
    ) -> None:
        wall_interval_ms = ""
        actual_fps = ""
        if self._last_draw_s is not None:
            interval_s = max(0.0, now_s - self._last_draw_s)
            wall_interval_ms = f"{interval_s * 1000.0:.3f}"
            if interval_s > 0.0:
                actual_fps = f"{1.0 / interval_s:.3f}"
        self._last_draw_s = now_s
        draw_latency_ms = (
            f"{(now_s - pending_tick_started_s) * 1000.0:.3f}"
            if pending_tick_started_s is not None
            else ""
        )
        self.record(
            "draw",
            event_number=event_number,
            source_timestamp_us=f"{source_timestamp_us:.3f}",
            wall_interval_ms=wall_interval_ms,
            actual_fps=actual_fps,
            draw_latency_ms=draw_latency_ms,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.record("session_end")
        self._file.flush()
        self._file.close()
        self._closed = True

    def flush(self) -> None:
        if not self._closed:
            self._file.flush()
            self._rows_since_flush = 0
