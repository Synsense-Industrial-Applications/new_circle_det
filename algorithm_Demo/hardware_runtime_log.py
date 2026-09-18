"""Temporary structured logging for the real Speck2f hardware demo.

The logger intentionally lives outside the detector implementation.  Removing
this file and the small calls in ``Demo_algorithm.py`` completely removes the diagnostic
instrumentation once the measurements are finished.

One run creates one directory containing reconstructable Layer-4 events,
per-update detector diagnostics, hardware batch timing, one-second rate
snapshots, markers, metadata, and a final summary.
"""

from __future__ import annotations

import atexit
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


EVENT_FIELDS = (
    "session_id",
    "wall_time",
    "elapsed_s",
    "batch_number",
    "layer4_sequence",
    "source_event_count",
    "timestamp_us",
    "raw_x64",
    "raw_y64",
    "feature",
    "x128",
    "y128",
    "direction",
)

UPDATE_FIELDS = (
    "session_id",
    "wall_time",
    "elapsed_s",
    "update_number",
    "source_event_count",
    "timestamp_us",
    "event_x",
    "event_y",
    "event_direction",
    "detection_ms",
    "has_candidate",
    "accepted",
    "failed_rules",
    "filter_decisions",
    "cx",
    "cy",
    "radius",
    "geometric_confidence",
    "full_confidence",
    "xiaoiron_confidence",
    "event_count",
    "inlier_count",
    "radial_inlier_ratio",
    "radial_mad_px",
    "angular_sectors",
    "quadrants",
    "direction_agreement",
    "hypotheses_tested",
    "occlusion_factor",
    "direction_factor",
    "evidence_strength_G",
    "direction_evidence_R",
    "direction_gap",
    "empty_visible_count",
    "side_sufficiency_q",
    "side_direction_evidence_qp",
    "sector_counts",
    "direction_weights",
    "sector_purity",
    "sector_visible",
)

BATCH_FIELDS = (
    "session_id",
    "wall_time",
    "elapsed_s",
    "batch_number",
    "raw_events",
    "layer4_events",
    "invalid_events",
    "detector_updates",
    "candidates",
    "accepted",
    "variant_type_counts",
    "dropped_message_count",
    "timestamp_nonmonotonic_count",
    "get_wait_ms",
    "processing_ms",
    "logging_ms",
    "reporting_ms",
    "process_cpu_ms",
    "main_process_cpu_ratio",
    "total_cycle_ms",
    "events_per_processing_s",
    "full_batch",
    "first_timestamp_us",
    "last_timestamp_us",
    "device_span_us",
    "cumulative_source_events",
    "cumulative_updates",
    "cumulative_accepted",
    "event_rows_waiting_for_flush",
)

RATE_FIELDS = (
    "session_id",
    "wall_time",
    "elapsed_s",
    "raw_events_per_s",
    "layer4_events_per_s",
    "batches_per_s",
    "updates_per_s",
    "last_batch_size",
    "full_batch_ratio",
    "average_detection_ms",
    "p95_recent256_detection_ms",
    "max_detection_ms",
    "cumulative_source_events",
    "cumulative_updates",
    "cumulative_candidates",
    "cumulative_accepted",
    "cumulative_rejected",
    "cumulative_no_candidate",
    "cumulative_invalid",
)

MARKER_FIELDS = (
    "session_id",
    "wall_time",
    "elapsed_s",
    "record_type",
    "name",
    "duration_ms",
    "event_number",
    "timestamp_us",
    "detail",
)


def _finite_or_blank(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _json_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()

    def normalise(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalise(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalise(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if hasattr(item, "item"):
            return normalise(item.item())
        return item

    return json.dumps(
        normalise(value), ensure_ascii=False, separators=(",", ":")
    )


class _CsvSink:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(fieldnames)

    def writerow(self, values: Sequence[Any]) -> None:
        self._writer.writerow(values)

    def writerows(self, rows: Iterable[Sequence[Any]]) -> None:
        self._writer.writerows(rows)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


class HardwareRuntimeLogger:
    """Buffered, dependency-free logger for ``algorithm_Demo/Demo_algorithm.py``."""

    def __init__(
        self,
        root: Path,
        *,
        event_buffer_rows: int = 8192,
        flush_interval_s: float = 1.0,
    ) -> None:
        started = datetime.now()
        self.session_id = started.strftime("%Y%m%d_%H%M%S_%f")
        self.root = Path(root).expanduser().resolve()
        self.session_dir = self.root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._started_perf = time.perf_counter()
        self._event_buffer_rows = max(1, int(event_buffer_rows))
        self._flush_interval_s = max(0.1, float(flush_interval_s))
        self._last_flush_perf = self._started_perf
        self._event_buffer: list[tuple[Any, ...]] = []
        self._closed = False
        self._summary: dict[str, Any] = {}
        self.event_rows = 0
        self.update_rows = 0
        self.batch_rows = 0
        self.rate_rows = 0
        self.marker_rows = 0
        self.flush_count = 0
        self.total_logging_call_ms = 0.0
        self._batch_number = 0
        self._layer4_sequence = 0
        self._previous_device_timestamp: Optional[float] = None

        self._events = _CsvSink(self.session_dir / "events.csv", EVENT_FIELDS)
        self._updates = _CsvSink(self.session_dir / "updates.csv", UPDATE_FIELDS)
        self._batches = _CsvSink(self.session_dir / "batches.csv", BATCH_FIELDS)
        self._rates = _CsvSink(self.session_dir / "rates.csv", RATE_FIELDS)
        self._markers = _CsvSink(self.session_dir / "markers.csv", MARKER_FIELDS)
        self._sinks = (
            self._events,
            self._updates,
            self._batches,
            self._rates,
            self._markers,
        )

        (self.root / "LATEST.txt").write_text(
            self.session_id + "\n", encoding="utf-8"
        )
        self._write_readme()
        self.record_metadata({})
        self.record_marker("session_start", "Demo_algorithm.py started")
        atexit.register(self.close)

    def _base(self) -> tuple[str, str, str]:
        now = datetime.now().isoformat(timespec="milliseconds")
        elapsed = time.perf_counter() - self._started_perf
        return self.session_id, now, f"{elapsed:.6f}"

    def _write_readme(self) -> None:
        text = "Speck2f hardware runtime log\n\n"
        text += "events.csv   - every valid decoded Layer-4 optical-flow event\n"
        text += "updates.csv  - every detector update, circle, confidence components and sector evidence\n"
        text += "batches.csv  - every hardware read cycle and its measured timing\n"
        text += "rates.csv    - one-second throughput/backlog snapshots\n"
        text += "markers.csv  - startup stages, errors, hits, replay and stop reason\n"
        text += "metadata.json - code parameters and machine/runtime information\n"
        text += "summary.json  - final counters and logger overhead\n\n"
        text += "For analysis, copy or zip this entire session directory.\n"
        (self.session_dir / "README.txt").write_text(text, encoding="utf-8")

    def record_metadata(self, metadata: Mapping[str, Any]) -> None:
        payload = {
            "session_id": self.session_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "event_buffer_rows": self._event_buffer_rows,
            "flush_interval_s": self._flush_interval_s,
            **dict(metadata),
        }
        (self.session_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def record_events(
        self,
        batch_number: int,
        rows: Iterable[Sequence[Any]],
    ) -> float:
        """Queue event payload rows; returns time spent in the logger in ms.

        Each payload is ``(layer4_sequence, source_event_count, timestamp_us,
        raw_x64, raw_y64, feature, x128, y128, direction)``.
        """

        started = time.perf_counter()
        base = self._base()
        added = 0
        for row in rows:
            self._event_buffer.append((*base, int(batch_number), *row))
            added += 1
        self.event_rows += added
        if len(self._event_buffer) >= self._event_buffer_rows:
            self._flush_event_buffer()
        self._flush_if_due()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_logging_call_ms += elapsed_ms
        return elapsed_ms

    def begin_batch(self) -> "HardwareBatchRecorder":
        self._batch_number += 1
        return HardwareBatchRecorder(self, self._batch_number)

    def _flush_event_buffer(self) -> None:
        if not self._event_buffer:
            return
        self._events.writerows(self._event_buffer)
        self._event_buffer.clear()

    def record_update(self, update: Any) -> float:
        started = time.perf_counter()
        detection = update.detection
        report = update.filter_report
        failed_rules = ""
        decisions = ""
        if report is not None:
            failed_rules = ";".join(item.rule_name for item in report.failures)
            decisions = _json_value(
                [
                    {
                        "rule": item.rule_name,
                        "passed": item.passed,
                        "actual": item.actual,
                        "expected": item.expected,
                    }
                    for item in report.decisions
                ]
            )

        row = {
            "update_number": update.update_number,
            "source_event_count": update.source_event_count,
            "timestamp_us": _finite_or_blank(update.event.t),
            "event_x": _finite_or_blank(update.event.x),
            "event_y": _finite_or_blank(update.event.y),
            "event_direction": update.event.c,
            "detection_ms": f"{update.detection_ms:.6f}",
            "has_candidate": int(detection is not None),
            "accepted": int(update.accepted),
            "failed_rules": failed_rules,
            "filter_decisions": decisions,
        }
        if detection is not None:
            row.update(
                {
                    "cx": detection.cx,
                    "cy": detection.cy,
                    "radius": detection.radius,
                    "geometric_confidence": detection.confidence,
                    "full_confidence": detection.full_confidence,
                    "xiaoiron_confidence": detection.xiaoiron_confidence,
                    "event_count": detection.event_count,
                    "inlier_count": detection.inlier_count,
                    "radial_inlier_ratio": detection.radial_inlier_ratio,
                    "radial_mad_px": detection.radial_mad,
                    "angular_sectors": detection.angular_sectors,
                    "quadrants": detection.quadrants,
                    "direction_agreement": detection.direction_agreement,
                    "hypotheses_tested": detection.hypotheses_tested,
                }
            )
            score = detection.xiaoiron
            if score is not None:
                row.update(
                    {
                        "occlusion_factor": score.occlusion_factor,
                        "direction_factor": score.direction_factor,
                        "evidence_strength_G": score.evidence_strength,
                        "direction_evidence_R": score.direction_evidence,
                        "direction_gap": score.direction_gap,
                        "empty_visible_count": score.empty_visible_count,
                        "side_sufficiency_q": _json_value(score.side_sufficiency),
                        "side_direction_evidence_qp": _json_value(
                            score.side_direction_evidence
                        ),
                        "sector_counts": _json_value(score.sector_counts),
                        "direction_weights": _json_value(score.direction_weights),
                        "sector_purity": _json_value(score.sector_purity),
                        "sector_visible": _json_value(score.sector_visible),
                    }
                )

        values = (*self._base(), *(row.get(name, "") for name in UPDATE_FIELDS[3:]))
        self._updates.writerow(values)
        self.update_rows += 1
        self._flush_if_due()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_logging_call_ms += elapsed_ms
        return elapsed_ms

    def record_batch(self, **values: Any) -> float:
        started = time.perf_counter()
        values.setdefault("event_rows_waiting_for_flush", len(self._event_buffer))
        row = (*self._base(), *(values.get(name, "") for name in BATCH_FIELDS[3:]))
        self._batches.writerow(row)
        self.batch_rows += 1
        self._flush_if_due()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_logging_call_ms += elapsed_ms
        return elapsed_ms

    def record_rate(self, **values: Any) -> float:
        started = time.perf_counter()
        row = (*self._base(), *(values.get(name, "") for name in RATE_FIELDS[3:]))
        self._rates.writerow(row)
        self.rate_rows += 1
        self._flush_if_due()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_logging_call_ms += elapsed_ms
        return elapsed_ms

    def record_rate_snapshot(self, reporter: Any) -> float:
        stats = reporter.pipeline.stats
        return self.record_rate(
            raw_events_per_s=reporter.raw_events_per_sec,
            layer4_events_per_s=reporter.layer4_events_per_sec,
            batches_per_s=reporter.batches_per_sec,
            updates_per_s=reporter.updates_per_sec,
            last_batch_size=reporter.last_batch_size,
            full_batch_ratio=reporter.full_batch_ratio,
            average_detection_ms=stats.average_detection_ms,
            p95_recent256_detection_ms=stats.p95_detection_ms,
            max_detection_ms=stats.max_detection_ms,
            cumulative_source_events=stats.source_events,
            cumulative_updates=stats.detector_updates,
            cumulative_candidates=stats.candidates,
            cumulative_accepted=stats.accepted,
            cumulative_rejected=stats.rejected,
            cumulative_no_candidate=stats.no_candidate,
            cumulative_invalid=stats.invalid_events,
        )

    def record_marker(
        self,
        name: str,
        detail: Any = "",
        *,
        record_type: str = "marker",
        duration_ms: Any = "",
        event_number: Any = "",
        timestamp_us: Any = "",
    ) -> float:
        if self._closed:
            return 0.0
        started = time.perf_counter()
        if detail and not isinstance(detail, str):
            detail = _json_value(detail)
        self._markers.writerow(
            (
                *self._base(),
                record_type,
                name,
                duration_ms,
                event_number,
                timestamp_us,
                detail,
            )
        )
        self.marker_rows += 1
        self._flush_if_due()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_logging_call_ms += elapsed_ms
        return elapsed_ms

    def record_stage(self, name: str, duration_ms: float, detail: str = "") -> float:
        return self.record_marker(
            name,
            detail,
            record_type="stage",
            duration_ms=f"{duration_ms:.6f}",
        )

    def record_invalid_event(self, error: BaseException, raw: Mapping[str, Any]) -> float:
        return self.record_marker(
            "invalid_layer4_event",
            {"error": f"{type(error).__name__}: {error}", "raw": dict(raw)},
            record_type="error",
        )

    def record_hit(self, trigger: Any) -> float:
        peak = trigger.peak
        return self.record_marker(
            "hit",
            {
                "hit_number": trigger.hit_number,
                "reason": trigger.reason,
                "detected_timestamp": trigger.detected_timestamp,
                "peak_update_number": peak.update_number,
                "peak_source_event_count": peak.source_event_count,
                "peak_timestamp": peak.timestamp,
                "cx": peak.cx,
                "cy": peak.cy,
                "radius": peak.radius,
                "smoothed_radius": peak.smoothed_radius,
                "confidence": peak.confidence,
                "radius_rise_px": trigger.radius_rise_px,
                "radius_fall_px": trigger.radius_fall_px,
            },
            event_number=peak.source_event_count,
            timestamp_us=peak.timestamp,
        )

    def record_replay(self, clip: Any, queued: bool, dropped_clips: int) -> float:
        return self.record_marker(
            "replay",
            {
                "hit_number": clip.trigger.hit_number,
                "queued": bool(queued),
                "complete": bool(clip.complete),
                "event_count": len(clip.events),
                "requested_start_timestamp": clip.requested_start_timestamp,
                "requested_end_timestamp": clip.requested_end_timestamp,
                "dropped_clips": int(dropped_clips),
            },
            timestamp_us=clip.trigger.peak.timestamp,
        )

    def record_summary(self, summary: Mapping[str, Any]) -> None:
        self._summary.update(dict(summary))

    def record_final_summary(
        self,
        pipeline: Any,
        reporter: Any,
        *,
        elapsed_s: float,
        stop_reason: str,
    ) -> None:
        stats = pipeline.stats
        self.record_summary(
            {
                "stop_reason": stop_reason,
                "elapsed_s": elapsed_s,
                "source_events": stats.source_events,
                "detector_updates": stats.detector_updates,
                "candidates": stats.candidates,
                "accepted": stats.accepted,
                "rejected": stats.rejected,
                "no_candidate": stats.no_candidate,
                "invalid_events": stats.invalid_events,
                "hits": reporter.hit_count,
                "replay_clips": reporter.replay_clips,
                "replay_events": reporter.replay_events,
                "replay_dropped": reporter.replay_dropped,
                "rejection_counts": dict(stats.rejection_counts),
                "average_detection_ms": stats.average_detection_ms,
                "p95_recent256_detection_ms": stats.p95_detection_ms,
                "max_detection_ms": stats.max_detection_ms,
            }
        )

    def _flush_if_due(self) -> None:
        now = time.perf_counter()
        if now - self._last_flush_perf < self._flush_interval_s:
            return
        self._flush_event_buffer()
        for sink in self._sinks:
            sink.flush()
        self._last_flush_perf = now
        self.flush_count += 1

    def flush(self) -> None:
        if self._closed:
            return
        self._flush_event_buffer()
        for sink in self._sinks:
            sink.flush()
        self._last_flush_perf = time.perf_counter()
        self.flush_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self.record_marker("session_end", "Demo_algorithm.py stopped")
        self.flush()
        elapsed_s = time.perf_counter() - self._started_perf
        summary = {
            **self._summary,
            "logger": {
                "session_id": self.session_id,
                "elapsed_s": elapsed_s,
                "event_rows": self.event_rows,
                "update_rows": self.update_rows,
                "batch_rows": self.batch_rows,
                "rate_rows": self.rate_rows,
                "marker_rows": self.marker_rows,
                "flush_count": self.flush_count,
                "total_logging_call_ms": self.total_logging_call_ms,
                "average_logging_call_us_per_event": (
                    1000.0 * self.total_logging_call_ms / max(1, self.event_rows)
                ),
            },
        }
        (self.session_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        for sink in self._sinks:
            sink.close()
        self._closed = True


class HardwareBatchRecorder:
    """One hardware read-cycle measurement with deferred CSV writes."""

    def __init__(self, logger: HardwareRuntimeLogger, batch_number: int) -> None:
        self.logger = logger
        self.batch_number = batch_number
        self.cycle_started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.get_wait_ms = 0.0
        self.processing_started: Optional[float] = None
        self.processing_ms = 0.0
        self.logging_ms = 0.0
        self.logging_finished: Optional[float] = None
        self.raw_event_count = 0
        self.layer4_event_count = 0
        self.invalid_event_count = 0
        self.dropped_message_count = 0
        self.timestamp_nonmonotonic_count = 0
        self.variant_type_counts: dict[str, int] = {}
        self.event_rows: list[tuple[Any, ...]] = []
        self.updates: list[Any] = []
        self.invalid_events: list[tuple[BaseException, dict[str, Any]]] = []
        self.first_timestamp_us: Optional[float] = None
        self.last_timestamp_us: Optional[float] = None
        self._processing_done = False

    def received(self, events: Sequence[Any]) -> None:
        """Call immediately after ``get_n_events`` returns."""

        self.get_wait_ms = (time.perf_counter() - self.cycle_started) * 1000.0
        self.raw_event_count = len(events)
        for event in events:
            event_type = type(event).__name__
            self.variant_type_counts[event_type] = (
                self.variant_type_counts.get(event_type, 0) + 1
            )
            if "drop" in event_type.lower():
                self.dropped_message_count += 1
        self.processing_started = time.perf_counter()

    @staticmethod
    def _raw_values(event: Any) -> dict[str, Any]:
        return {
            "x": getattr(event, "x", None),
            "y": getattr(event, "y", None),
            "feature": getattr(event, "feature", None),
            "timestamp": getattr(event, "timestamp", None),
        }

    def record_invalid(self, event: Any, error: BaseException) -> None:
        self.logger._layer4_sequence += 1
        self.layer4_event_count += 1
        self.invalid_event_count += 1
        self.invalid_events.append((error, self._raw_values(event)))

    def record_event(
        self,
        raw_event: Any,
        flow_event: Any,
        source_event_count: int,
    ) -> None:
        self.logger._layer4_sequence += 1
        self.layer4_event_count += 1
        timestamp = float(flow_event.t)
        previous = self.logger._previous_device_timestamp
        if previous is not None and timestamp < previous:
            self.timestamp_nonmonotonic_count += 1
        self.logger._previous_device_timestamp = timestamp
        if self.first_timestamp_us is None:
            self.first_timestamp_us = timestamp
        self.last_timestamp_us = timestamp
        raw = self._raw_values(raw_event)
        self.event_rows.append(
            (
                self.logger._layer4_sequence,
                source_event_count,
                timestamp,
                raw["x"],
                raw["y"],
                raw["feature"],
                flow_event.x,
                flow_event.y,
                flow_event.c,
            )
        )

    def record_update(self, update: Any) -> None:
        self.updates.append(update)

    def processing_done(self) -> None:
        if self._processing_done:
            return
        now = time.perf_counter()
        started = self.processing_started or self.cycle_started
        self.processing_ms = (now - started) * 1000.0
        logging_started = time.perf_counter()
        for error, raw in self.invalid_events:
            self.logger.record_invalid_event(error, raw)
        for update in self.updates:
            self.logger.record_update(update)
        self.logger.record_events(self.batch_number, self.event_rows)
        self.logging_ms = (time.perf_counter() - logging_started) * 1000.0
        self.logging_finished = time.perf_counter()
        self._processing_done = True

    def finish(self, stats: Any) -> None:
        self.processing_done()
        now = time.perf_counter()
        reporting_ms = (
            (now - self.logging_finished) * 1000.0
            if self.logging_finished is not None
            else 0.0
        )
        total_cycle_ms = (now - self.cycle_started) * 1000.0
        process_cpu_ms = (time.process_time() - self.cpu_started) * 1000.0
        candidates = sum(update.detection is not None for update in self.updates)
        accepted = sum(update.accepted for update in self.updates)
        device_span_us: Any = ""
        if self.first_timestamp_us is not None and self.last_timestamp_us is not None:
            device_span_us = self.last_timestamp_us - self.first_timestamp_us
        self.logger.record_batch(
            batch_number=self.batch_number,
            raw_events=self.raw_event_count,
            layer4_events=self.layer4_event_count,
            invalid_events=self.invalid_event_count,
            detector_updates=len(self.updates),
            candidates=candidates,
            accepted=accepted,
            variant_type_counts=_json_value(self.variant_type_counts),
            dropped_message_count=self.dropped_message_count,
            timestamp_nonmonotonic_count=self.timestamp_nonmonotonic_count,
            get_wait_ms=f"{self.get_wait_ms:.6f}",
            processing_ms=f"{self.processing_ms:.6f}",
            logging_ms=f"{self.logging_ms:.6f}",
            reporting_ms=f"{reporting_ms:.6f}",
            process_cpu_ms=f"{process_cpu_ms:.6f}",
            main_process_cpu_ratio=(
                process_cpu_ms / total_cycle_ms if total_cycle_ms > 0.0 else ""
            ),
            total_cycle_ms=f"{total_cycle_ms:.6f}",
            events_per_processing_s=(
                self.layer4_event_count * 1000.0 / self.processing_ms
                if self.processing_ms > 0.0
                else ""
            ),
            full_batch=int(self.raw_event_count >= 512),
            first_timestamp_us=(
                self.first_timestamp_us
                if self.first_timestamp_us is not None
                else ""
            ),
            last_timestamp_us=(
                self.last_timestamp_us if self.last_timestamp_us is not None else ""
            ),
            device_span_us=device_span_us,
            cumulative_source_events=stats.source_events,
            cumulative_updates=stats.detector_updates,
            cumulative_accepted=stats.accepted,
        )
