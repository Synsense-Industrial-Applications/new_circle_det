"""Real-time circle-detection runtime for the Speck2f Layer-4 stream.

The SNN network, layer wiring, and hardware visualization routes live in
``Demo_SNN.py``.  This module owns only the downstream circle algorithm,
terminal reporting, hit detection/replay, runtime logging, and executable loop.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
import samna


# Demo_algorithm.py lives one directory below the reusable circle_detection package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_detection import AdaptiveDetectorConfig, XiaoironConfig  # noqa: E402
from layer4_layout import LAYER4_FEATURE_COUNT  # noqa: E402
from circle_runtime import (  # noqa: E402
    CircleDetectionPipeline,
    CircleFilterConfig,
    DEFAULT_CIRCLE_FILTERS,
    DIRECTION_ANGLES_DEG,
    FEATURE_TO_DIRECTION,
    decode_layer4_event,
)
from hit_replay import (  # noqa: E402
    HitReplayConfig,
    HitReplayViewer,
    RadiusPeakHitDetector,
    SlidingEventWindow,
    format_hit_trigger,
)
from hardware_runtime_log import HardwareRuntimeLogger  # noqa: E402
from Demo_SNN import (  # noqa: E402
    config,
    configure_cnn_pipeline,
    layer_4,
    open_speck2f_dev_kit,
    visualize_layer,
)


# ===========================================================================
# 可调参数：检测、输出过滤和终端显示都集中在这里
# ===========================================================================

# 输出规则。半径是闭区间 [35, 41]；圆心范围是严格开区间。
CIRCLE_FILTER_CONFIG = CircleFilterConfig(
    confidence_attribute="xiaoiron_confidence",
    min_confidence=0.30,
    min_radius_px=35.0,
    max_radius_px=41.0,
    min_center_x_px=30.0,
    max_center_x_px=90.0,
    min_center_y_px=40.0,
    max_center_y_px=100.0,
)

# 小铁置信度参数（名称与离线播放器滑块一致）。
# 修改后重启 Demo_algorithm.py 生效；硬件 Demo 本身不显示 GUI 滑块。
SCORE_WINDOW_EVENTS = 230  # score N
XIAOIRON_CONFIDENCE_CONFIG = XiaoironConfig(
    occlusion_gain=0.51,              # alpha
    direction_penalty_lambda=0.85,    # lambda
    min_direction_events=5,           # N_min
    min_direction_weight=1.75,        # W_min
    min_other_sectors=6,              # K_min
    min_sector_weight=1.0,            # sector W
)

# 最新自适应算法的主要吞吐/精度参数。候选半径范围保留少量余量，
# 最终 [35, 41] 由上面的具名 radius 输出规则执行。
CIRCLE_DETECTOR_CONFIG = AdaptiveDetectorConfig(
    window_events=SCORE_WINDOW_EVENTS,
    decay_events=120.0,
    min_events=80,
    hypotheses=64,
    refine_candidates=10,
    refine_iterations=4,
    update_interval_events=6,
    min_radius_px=34.0,
    max_radius_px=42.0,
    xiaoiron=XIAOIRON_CONFIDENCE_CONFIG,
)

# 新增过滤条件时，只需实现一个 CircleFilterRule 并追加到这个元组；
# CircleDetectionPipeline 主流程和拒绝统计无需修改。
CIRCLE_FILTER_RULES = DEFAULT_CIRCLE_FILTERS


# 击球时机和回放。半径先上升后回落时取局部峰值为击球时刻；如果半径持续
# 上升后连续丢圆，也可以把最后一个有效圆作为击球时刻。时间单位为微秒。
HIT_REPLAY_CONFIG = HitReplayConfig(
    enabled=True,
    cue_x_px=68.0,
    cue_y_px=83.0,
    timing_required_filter_rules=("radius", "center_x", "center_y"),
    min_timing_confidence=0.15,
    radius_ema_alpha=0.45,
    rise_window_samples=24,
    fall_confirm_samples=3,
    max_peak_age_samples=30,
    min_radius_rise_px=0.80,
    min_radius_fall_px=0.55,
    min_rising_fraction=0.60,
    min_falling_fraction=0.67,
    min_peak_radius_px=35.0,
    detect_rise_then_lost=True,
    lost_confirm_updates=3,
    lost_min_radius_px=38.0,
    lost_min_radius_rise_px=1.00,
    cooldown_us=800_000.0,
    pre_hit_us=150_000.0,
    post_hit_us=200_000.0,
    buffer_duration_us=700_000.0,
    max_buffer_events=60_000,
    replay_speed=0.20,
    replay_fps=30.0,
    event_trail_us=35_000.0,
    final_hold_sec=1.5,
    replay_queue_size=2,
)


@dataclass(frozen=True)
class TerminalConfig:
    status_interval_sec: float = 1.0
    rate_interval_sec: float = 1.0
    accepted_output_interval_sec: float = 0.10


TERMINAL_CONFIG = TerminalConfig()

# 临时实机诊断日志；分析完成后删除日志模块和 runtime_logger 钩子即可。
RUNTIME_LOG_ROOT = PROJECT_ROOT / "runtime_logs"
RUNTIME_LOG_EVENT_BUFFER_ROWS = 8192


# ===========================================================================
# 终端显示与运行状态
# ===========================================================================

def _confidence_value(detection):
    try:
        return float(getattr(
            detection,
            CIRCLE_FILTER_CONFIG.confidence_attribute,
        ))
    except (AttributeError, TypeError, ValueError):
        return float("nan")


def _hit_timing_detection(update):
    """Use stable geometry for timing even if final confidence briefly dips."""

    detection = update.detection
    report = update.filter_report
    if detection is None or report is None:
        return None
    decisions = {decision.rule_name: decision.passed for decision in report.decisions}
    if not all(
        decisions.get(rule_name, False)
        for rule_name in HIT_REPLAY_CONFIG.timing_required_filter_rules
    ):
        return None
    confidence = _confidence_value(detection)
    if (
        not np.isfinite(confidence)
        or confidence < HIT_REPLAY_CONFIG.min_timing_confidence
    ):
        return None
    return detection


def format_output_line(update):
    """Stable machine-readable line for an accepted circle only."""

    detection = update.detection
    if detection is None or not update.accepted:
        raise ValueError("format_output_line requires an accepted detection")
    return (
        f"[CIRCLE OUTPUT] update={update.update_number} "
        f"event={update.source_event_count} t_us={int(detection.timestamp)} "
        f"cx={detection.cx:.3f} cy={detection.cy:.3f} "
        f"radius={detection.radius:.3f} "
        f"{CIRCLE_FILTER_CONFIG.confidence_attribute}="
        f"{_confidence_value(detection):.4f}"
    )


class TerminalReporter:
    """Persistent circle outputs plus rate-limited diagnostic snapshots."""

    def __init__(self, pipeline, config=TERMINAL_CONFIG):
        self.pipeline = pipeline
        self.config = config
        self.last_update = None
        self.last_error = ""
        self.raw_events_per_sec = 0.0
        self.layer4_events_per_sec = 0.0
        self.batches_per_sec = 0.0
        self.updates_per_sec = 0.0
        self.last_batch_size = 0
        self.full_batch_ratio = 0.0
        self._last_refresh = 0.0
        self._last_output = 0.0
        self._pending_output = None
        self.output_lines_emitted = 0
        self.output_updates_coalesced = 0
        self.hit_count = 0
        self.replay_clips = 0
        self.replay_events = 0
        self.replay_dropped = 0
        self.last_hit_line = "HIT waiting for a radius peak"
        self.last_replay_line = "REPLAY waiting"

    def record_update(self, update):
        self.last_update = update
        if update.accepted:
            if self._pending_output is not None:
                self.output_updates_coalesced += 1
            self._pending_output = update
            self._emit_pending_output()

    def _emit_pending_output(self, force=False):
        if self._pending_output is None:
            return
        now = time.monotonic()
        interval = max(0.0, self.config.accepted_output_interval_sec)
        if not force and interval > 0.0 and now - self._last_output < interval:
            return
        print(format_output_line(self._pending_output))
        self._pending_output = None
        self._last_output = now
        self.output_lines_emitted += 1

    def flush(self):
        self._emit_pending_output(force=True)
        sys.stdout.flush()

    def record_invalid_event(self, error):
        self.last_error = str(error)

    def record_hit(self, trigger):
        self.hit_count += 1
        self.last_hit_line = format_hit_trigger(trigger)
        print(self.last_hit_line, flush=True)

    def record_replay(self, clip, queued, dropped_clips):
        self.replay_clips += int(bool(queued))
        self.replay_events += len(clip.events)
        self.replay_dropped = int(dropped_clips)
        state = "queued" if queued else "viewer_unavailable"
        completeness = "complete" if clip.complete else "partial"
        self.last_replay_line = (
            f"REPLAY hit={clip.trigger.hit_number} state={state} "
            f"clip={completeness} events={len(clip.events):,} "
            f"range=[-{HIT_REPLAY_CONFIG.pre_hit_us/1000:g}, "
            f"+{HIT_REPLAY_CONFIG.post_hit_us/1000:g}]ms "
            f"speed={HIT_REPLAY_CONFIG.replay_speed:g}x "
            f"dropped={self.replay_dropped}"
        )
        print(f"[HIT {self.last_replay_line}]", flush=True)

    def set_rates(
        self,
        *,
        raw_events_per_sec,
        layer4_events_per_sec,
        batches_per_sec,
        updates_per_sec,
        last_batch_size,
        full_batch_ratio,
    ):
        self.raw_events_per_sec = raw_events_per_sec
        self.layer4_events_per_sec = layer4_events_per_sec
        self.batches_per_sec = batches_per_sec
        self.updates_per_sec = updates_per_sec
        self.last_batch_size = last_batch_size
        self.full_batch_ratio = full_batch_ratio

    def _lines(self):
        stats = self.pipeline.stats
        detector_config = self.pipeline.detector_config
        window_fill = min(stats.source_events, detector_config.window_events)
        state_line = (
            "STATE "
            f"layer4_events={stats.source_events:,} "
            f"window={window_fill}/{detector_config.window_events} "
            f"updates={stats.detector_updates:,} "
            f"invalid={stats.invalid_events:,}"
        )

        update = self.last_update
        if update is None:
            candidate_line = (
                "CANDIDATE waiting: the detector needs "
                f"{detector_config.min_events} recent events"
            )
            quality_line = "QUALITY waiting for the first detector update"
            output_line = "OUTPUT none"
        elif update.detection is None:
            if update.source_event_count < detector_config.min_events:
                candidate_line = (
                    f"CANDIDATE warming update={update.update_number} "
                    f"event={update.source_event_count}/"
                    f"{detector_config.min_events} t_us={int(update.event.t)}"
                )
                quality_line = (
                    f"QUALITY detect={update.detection_ms:.3f}ms "
                    "(collecting the minimum event window)"
                )
            else:
                candidate_line = (
                    f"CANDIDATE none update={update.update_number} "
                    f"event={update.source_event_count} "
                    f"t_us={int(update.event.t)}"
                )
                quality_line = (
                    f"QUALITY detect={update.detection_ms:.3f}ms "
                    "(no geometrically valid circle)"
                )
            output_line = "OUTPUT none"
        else:
            detection = update.detection
            xconfidence = _confidence_value(detection)
            candidate_line = (
                f"CANDIDATE update={update.update_number} "
                f"event={update.source_event_count} t_us={int(detection.timestamp)} "
                f"center=({detection.cx:.3f},{detection.cy:.3f}) "
                f"r={detection.radius:.3f} "
                f"xconf={xconfidence:.4f} full={detection.full_confidence:.4f} "
                f"geom={detection.confidence:.4f}"
            )
            quality_line = (
                f"QUALITY inliers={detection.inlier_count}/{detection.event_count} "
                f"ratio={detection.radial_inlier_ratio:.3f} "
                f"MAD={detection.radial_mad:.3f}px "
                f"sectors={detection.angular_sectors} "
                f"quadrants={detection.quadrants} "
                f"direction={detection.direction_agreement:.3f} "
                f"hypotheses={detection.hypotheses_tested} "
                f"detect={update.detection_ms:.3f}ms"
            )
            if update.accepted:
                output_line = (
                    "OUTPUT ACCEPT "
                    f"cx={detection.cx:.3f} cy={detection.cy:.3f} "
                    f"r={detection.radius:.3f} conf={xconfidence:.4f}"
                )
            else:
                failures = (
                    update.filter_report.failure_summary()
                    if update.filter_report is not None
                    else "filter report unavailable"
                )
                output_line = f"OUTPUT REJECT {failures}"

        rejection_counts = " ".join(
            f"{rule.name}={stats.rejection_counts.get(rule.name, 0)}"
            for rule in self.pipeline.filter_chain.rules
        )
        counter_line = (
            f"COUNTERS candidates={stats.candidates:,} "
            f"accepted={stats.accepted:,} rejected={stats.rejected:,} "
            f"none={stats.no_candidate:,} "
            f"terminal_outputs={self.output_lines_emitted:,} "
            f"coalesced={self.output_updates_coalesced:,} "
            f"reject_by_rule[{rejection_counts}]"
        )
        performance_line = (
            f"PERF raw={self.raw_events_per_sec:,.1f}/s "
            f"layer4={self.layer4_events_per_sec:,.1f}/s "
            f"batches={self.batches_per_sec:.1f}/s "
            f"updates={self.updates_per_sec:.1f}/s "
            f"detect_ms(avg_all/p95_recent256/max_all)="
            f"{stats.average_detection_ms:.3f}/"
            f"{stats.p95_detection_ms:.3f}/"
            f"{stats.max_detection_ms:.3f} "
            f"last_batch={self.last_batch_size} "
            f"full_batch={self.full_batch_ratio:.0%}"
        )
        if self.full_batch_ratio >= 0.50:
            performance_line += (
                " WARNING=possible_backlog; increase "
                "CIRCLE_DETECTOR_CONFIG.update_interval_events if persistent"
            )
        error_line = f"LAST ERROR {self.last_error}" if self.last_error else ""
        return [
            state_line,
            candidate_line,
            quality_line,
            output_line,
            counter_line,
            self.last_hit_line,
            self.last_replay_line,
            performance_line,
            error_line,
        ]

    def refresh(self, force=False):
        now = time.monotonic()
        self._emit_pending_output()
        if (
            not force
            and now - self._last_refresh < self.config.status_interval_sec
        ):
            return

        lines = [line for line in self._lines() if line]
        print("\n".join(lines), flush=True)
        self._last_refresh = now


def print_runtime_configuration():
    filters = CIRCLE_FILTER_CONFIG
    detector = CIRCLE_DETECTOR_CONFIG
    print("=" * 88)
    print("Speck2f latest adaptive circle detector - hardware realtime")
    print("=" * 88)
    print(
        "Algorithm: "
        f"window={detector.window_events} events, "
        f"min_events={detector.min_events}, "
        f"update_every={detector.update_interval_events} events, "
        f"hypotheses={detector.hypotheses}, "
        f"refine={detector.refine_candidates}x{detector.refine_iterations}, "
        f"candidate_radius=[{detector.min_radius_px:g}, "
        f"{detector.max_radius_px:g}]px"
    )
    print(
        "Directions: "
        f"feature->code={FEATURE_TO_DIRECTION}, angles={DIRECTION_ANGLES_DEG}"
    )
    print(
        "Output confidence: "
        f"{filters.confidence_attribute} >= {filters.min_confidence:.3f}"
    )
    print(
        "Output geometry: "
        f"radius=[{filters.min_radius_px:g}, {filters.max_radius_px:g}]px, "
        f"cx=({filters.min_center_x_px:g}, {filters.max_center_x_px:g})px, "
        f"cy=({filters.min_center_y_px:g}, {filters.max_center_y_px:g})px"
    )
    print(
        "Filter chain: "
        + " -> ".join(rule.name for rule in CIRCLE_FILTER_RULES)
    )
    print(
        "Only OUTPUT ACCEPT / [CIRCLE OUTPUT] values are emitted circles; "
        "CANDIDATE lines are diagnostics."
    )
    print(
        "Terminal: persistent status every "
        f"{TERMINAL_CONFIG.status_interval_sec:g}s; "
        "accepted output interval="
        f"{TERMINAL_CONFIG.accepted_output_interval_sec:g}s "
        "(0 means every accepted update)"
    )
    hit = HIT_REPLAY_CONFIG
    print(
        "Hit timing: radius EMA alpha="
        f"{hit.radius_ema_alpha:g}, rise={hit.min_radius_rise_px:g}px/"
        f"{hit.rise_window_samples} samples, "
        f"fall={hit.min_radius_fall_px:g}px/"
        f"{hit.fall_confirm_samples} samples, "
        f"timing_conf>={hit.min_timing_confidence:g}, "
        f"lost_path={hit.detect_rise_then_lost}, "
        f"cooldown={hit.cooldown_us/1000:g}ms"
    )
    print(
        "Hit replay: "
        f"enabled={hit.enabled}, pre={hit.pre_hit_us/1000:g}ms, "
        f"post={hit.post_hit_us/1000:g}ms, "
        f"speed={hit.replay_speed:g}x, cue=({hit.cue_x_px:g}, {hit.cue_y_px:g})"
    )
    print("=" * 88)


def print_final_summary(pipeline, reporter, elapsed_sec):
    stats = pipeline.stats
    print("=" * 88)
    print("FINAL SUMMARY")
    print(
        f"elapsed={elapsed_sec:.3f}s layer4_events={stats.source_events:,} "
        f"updates={stats.detector_updates:,} candidates={stats.candidates:,} "
        f"accepted={stats.accepted:,} rejected={stats.rejected:,} "
        f"none={stats.no_candidate:,} invalid={stats.invalid_events:,} "
        f"hits={reporter.hit_count:,} replay_clips={reporter.replay_clips:,} "
        f"replay_events={reporter.replay_events:,} "
        f"replay_dropped={reporter.replay_dropped:,} "
        f"terminal_outputs={reporter.output_lines_emitted:,} "
        f"coalesced={reporter.output_updates_coalesced:,}"
    )
    print(
        "reject_by_rule: "
        + ", ".join(
            f"{rule.name}={stats.rejection_counts.get(rule.name, 0)}"
            for rule in pipeline.filter_chain.rules
        )
    )
    print(
        "detect_ms avg_all/p95_recent256/max_all="
        f"{stats.average_detection_ms:.3f}/"
        f"{stats.p95_detection_ms:.3f}/"
        f"{stats.max_detection_ms:.3f}"
    )
    print("=" * 88)


# ===========================================================================
# 主流程
# ===========================================================================

def _run_demo(runtime_logger):
    print_runtime_configuration()

    print("Configuring CNN pipeline...")
    configure_cnn_pipeline()

    print("Opening Speck2f device...")
    dk = open_speck2f_dev_kit()

    input_graph = samna.graph.EventFilterGraph()
    input_buffer = samna.BasicSourceNode_speck2f_event_input_event()
    input_graph.sequential([input_buffer, dk.get_model_sink_node()])
    input_graph.start()

    dk.get_model().apply_configuration(config)

    # Creating this route starts device output.  Keep the returned source alive.
    device_input_route = samna.graph.source_to(dk.get_model_sink_node())
    event_buffer = samna.graph.sink_from(dk.get_model_source_node())

    io_module = dk.get_io_module()
    io_module.set_slow_clk_rate(32)
    io_module.set_slow_clk(True)
    io_module.set_in_out_interface_clk_rate(1_000_000)
    dk.get_power_module().set_vdd_io(3.3)

    stopwatch = dk.get_stop_watch()
    stopwatch.reset()
    stopwatch.start()

    print("Opening samnagui Layer-4 activity view...")
    viz_graph, viz_gui = visualize_layer(dk, layer_4)

    pipeline = CircleDetectionPipeline(
        CIRCLE_DETECTOR_CONFIG,
        CIRCLE_FILTER_CONFIG,
        CIRCLE_FILTER_RULES,
    )
    reporter = TerminalReporter(pipeline)
    hit_detector = RadiusPeakHitDetector(HIT_REPLAY_CONFIG)
    hit_event_window = SlidingEventWindow(HIT_REPLAY_CONFIG)
    hit_replay_viewer = HitReplayViewer(HIT_REPLAY_CONFIG)
    replay_started = hit_replay_viewer.start()
    print(
        "Hit replay viewer: "
        + ("started" if replay_started else "disabled or unavailable")
    )

    def dispatch_ready_replays():
        for clip in hit_event_window.pop_ready():
            queued = hit_replay_viewer.submit(clip)
            reporter.record_replay(
                clip,
                queued=queued,
                dropped_clips=hit_replay_viewer.dropped_clips,
            )
            runtime_logger.record_replay(
                clip, queued, hit_replay_viewer.dropped_clips
            )

    # Discard events left over from configuration.
    event_buffer.get_events()

    started = time.monotonic()
    rate_started = started
    raw_events_interval = 0
    layer4_events_interval = 0
    batches_interval = 0
    full_batches_interval = 0
    last_batch_size = 0
    previous_update_count = 0
    stop_reason = "running"

    print("Detector ready. Press Ctrl+C to stop.")
    runtime_logger.record_marker("detector_ready", "hardware event loop started")

    try:
        while True:
            batch_log = runtime_logger.begin_batch()
            events = event_buffer.get_n_events(n=512, timeout=10)
            batch_log.received(events)

            if events:
                last_batch_size = len(events)
                raw_events_interval += len(events)
                batches_interval += 1
                if len(events) >= 512:
                    full_batches_interval += 1

                for event in events:
                    # The model source is a variant stream and can also contain
                    # DVS, dropped-event or register-response messages.
                    if getattr(event, "layer", None) != layer_4:
                        continue
                    layer4_events_interval += 1
                    try:
                        flow_event = decode_layer4_event(
                            getattr(event, "x"),
                            getattr(event, "y"),
                            getattr(event, "feature"),
                            getattr(event, "timestamp"),
                        )
                    except (AttributeError, TypeError, ValueError) as error:
                        pipeline.stats.invalid_events += 1
                        reporter.record_invalid_event(error)
                        batch_log.record_invalid(event, error)
                        continue
                    if HIT_REPLAY_CONFIG.enabled:
                        hit_event_window.push(flow_event)
                        dispatch_ready_replays()
                    update = pipeline.process_flow_event(flow_event)
                    if update is not None:
                        batch_log.record_update(update)
                        reporter.record_update(update)
                        trigger = None
                        if HIT_REPLAY_CONFIG.enabled:
                            detection = _hit_timing_detection(update)
                            if detection is None:
                                trigger = hit_detector.observe_missing(
                                    timestamp=flow_event.t
                                )
                            else:
                                trigger = hit_detector.observe_circle(
                                    update_number=update.update_number,
                                    source_event_count=update.source_event_count,
                                    timestamp=detection.timestamp,
                                    cx=detection.cx,
                                    cy=detection.cy,
                                    radius=detection.radius,
                                    confidence=_confidence_value(detection),
                                )
                            if trigger is not None:
                                reporter.record_hit(trigger)
                                runtime_logger.record_hit(trigger)
                                hit_event_window.arm(trigger)
                                dispatch_ready_replays()
                    batch_log.record_event(
                        event, flow_event, pipeline.stats.source_events
                    )

            batch_log.processing_done()

            now = time.monotonic()
            rate_elapsed = now - rate_started
            if rate_elapsed >= TERMINAL_CONFIG.rate_interval_sec:
                update_count = pipeline.stats.detector_updates
                reporter.set_rates(
                    raw_events_per_sec=raw_events_interval / rate_elapsed,
                    layer4_events_per_sec=layer4_events_interval / rate_elapsed,
                    batches_per_sec=batches_interval / rate_elapsed,
                    updates_per_sec=(
                        update_count - previous_update_count
                    ) / rate_elapsed,
                    last_batch_size=last_batch_size,
                    full_batch_ratio=(
                        full_batches_interval / max(1, batches_interval)
                    ),
                )
                runtime_logger.record_rate_snapshot(reporter)
                raw_events_interval = 0
                layer4_events_interval = 0
                batches_interval = 0
                full_batches_interval = 0
                previous_update_count = update_count
                rate_started = now

            reporter.refresh()
            batch_log.finish(pipeline.stats)

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        runtime_logger.record_marker("stop_requested", "Ctrl+C")

    except BaseException as error:
        stop_reason = f"error:{type(error).__name__}"
        runtime_logger.record_marker(
            "event_loop_error",
            f"{type(error).__name__}: {error}",
            record_type="error",
        )
        runtime_logger.record_final_summary(
            pipeline,
            reporter,
            elapsed_s=time.monotonic() - started,
            stop_reason=stop_reason,
        )
        raise

    finally:
        reporter.flush()
        if HIT_REPLAY_CONFIG.enabled:
            for clip in hit_event_window.flush():
                queued = hit_replay_viewer.submit(clip)
                reporter.record_replay(
                    clip,
                    queued=queued,
                    dropped_clips=hit_replay_viewer.dropped_clips,
                )
                runtime_logger.record_replay(
                    clip, queued, hit_replay_viewer.dropped_clips
                )
        hit_replay_viewer.close(timeout_sec=5.0)
        try:
            input_graph.stop()
        except Exception:
            pass
        try:
            viz_graph.stop()
        except Exception:
            pass
        try:
            viz_gui.terminate()
            viz_gui.join(timeout=2)
        except Exception:
            pass
        # Keep this reference intentional and explicit until all graphs stop.
        _ = device_input_route

    elapsed_sec = time.monotonic() - started
    runtime_logger.record_final_summary(
        pipeline, reporter, elapsed_s=elapsed_sec, stop_reason=stop_reason
    )
    print_final_summary(pipeline, reporter, elapsed_sec)
    print("Speck2f adaptive circle detector stopped.")


def main():
    runtime_logger = HardwareRuntimeLogger(
        RUNTIME_LOG_ROOT,
        event_buffer_rows=RUNTIME_LOG_EVENT_BUFFER_ROWS,
    )
    print(f"Runtime log session: {runtime_logger.session_dir}")
    runtime_logger.record_metadata(
        {
            "entrypoint": str(Path(__file__).resolve()),
            "argv": sys.argv,
            "circle_filter_config": asdict(CIRCLE_FILTER_CONFIG),
            "xiaoiron_confidence_config": asdict(XIAOIRON_CONFIDENCE_CONFIG),
            "circle_detector_config": asdict(CIRCLE_DETECTOR_CONFIG),
            "circle_filter_rules": [rule.name for rule in CIRCLE_FILTER_RULES],
            "hit_replay_config": asdict(HIT_REPLAY_CONFIG),
            "terminal_config": asdict(TERMINAL_CONFIG),
            "layer4": layer_4,
            "layer4_feature_count": LAYER4_FEATURE_COUNT,
            "feature_to_direction": FEATURE_TO_DIRECTION,
            "direction_angles_deg": DIRECTION_ANGLES_DEG,
        }
    )
    try:
        _run_demo(runtime_logger)
    except BaseException as error:
        runtime_logger.record_marker(
            "fatal_error",
            f"{type(error).__name__}: {error}",
            record_type="error",
        )
        raise
    finally:
        runtime_logger.close()


if __name__ == "__main__":
    main()
