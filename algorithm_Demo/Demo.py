"""Speck2f hardware demo using the latest adaptive circle detector.

Data flow:
    DVS -> CNN Layer-4 (64x64x8) -> decode to (x128, y128, direction)
        -> adaptive three-point circle consensus -> named output filters

Only circles that pass every registered output filter are emitted.  Rejected
candidates remain visible in terminal diagnostics so parameters can be tuned.
"""

from dataclasses import asdict, dataclass
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import samna
import samnagui

from speck_tools import ChannelHelper


# Demo.py lives one directory below the reusable circle_detection package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circle_detection import AdaptiveDetectorConfig, XiaoironConfig  # noqa: E402
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
# 修改后重启 Demo.py 生效；硬件 Demo 本身不显示 GUI 滑块。
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
# Speck2f CNN 配置 (from test_online_ring_V1_3.py)
# ===========================================================================

ch = ChannelHelper(input_channel=1, input_size=128, output_size=64)

# ── 构建卷积核 ──
w_128 = np.zeros((2, 2, 5, 5))
w_128[1, 1, range(5), range(5)] = [-1, -1, 1, -2, -1]
w_128[1, 0, range(5), range(5)] = [0, 0, -1, 2, 0]
w_128[0, 1, range(5), range(5)] = [-1, -2, 1, -1, -1]
w_128[0, 0, range(5), range(5)] = [0, 2, -1, 0, 0]
w_1_to_2 = np.concatenate([np.concatenate([ch.get_kernel(w_128[i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)
w_1_to_2_t = np.concatenate([np.concatenate([ch.get_kernel(w_128[:, :, ::-1, :][i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)

w_128 = np.zeros((2, 1, 5, 5))
w_128[1, 0, range(5), range(5)] = [-1, 0, 0, -2, -1]
w_128[0, 0, range(5), range(5)] = [-1, -2, 0, 0, -1]
w_0_to_3 = np.concatenate([np.concatenate([ch.get_kernel(w_128[i, j]) for j in range(1)], axis=1) for i in range(2)], axis=0)
w_0_to_3_t = np.concatenate([np.concatenate([ch.get_kernel(w_128[:, :, ::-1, :][i, j]) for j in range(1)], axis=1) for i in range(2)], axis=0)

w_128 = np.zeros((2, 2, 5, 5))
w_128[1, 1, range(5), range(5)] = [0, 1, 2, 0, 0]
w_128[1, 0, range(5), range(5)] = [0, 0, -1, 0, 0]
w_128[0, 1, range(5), range(5)] = [0, 0, -1, 0, 0]
w_128[0, 0, range(5), range(5)] = [0, 0, 2, 1, 0]
w_2_to_3 = np.concatenate([np.concatenate([ch.get_kernel(w_128[i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)
w_2_to_3_t = np.concatenate([np.concatenate([ch.get_kernel(w_128[:, :, ::-1, :][i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)

M = 5
w_128 = np.zeros((2, 2, M - 2, M - 2))
w_128[1, 1, range(M - 2), range(M - 2)] = [-1] * (M - 4) + [-2, 0]
w_128[0, 0, range(M - 2), range(M - 2)] = [0, -2] + [-1] * (M - 4)
w_2_to_4 = np.concatenate([np.concatenate([ch.get_kernel(w_128[i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)
w_2_to_4_t = np.concatenate([np.concatenate([ch.get_kernel(w_128[:, :, ::-1, :][i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)

w_128 = np.zeros((2, 2, M - 2, M - 2))
w_128[1, 1, range(M - 2), range(M - 2)] = [1] * (M - 3) + [2]
w_128[0, 0, range(M - 2), range(M - 2)] = [2] + [1] * (M - 3)
w_3_to_4 = np.concatenate([np.concatenate([ch.get_kernel(w_128[i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)
w_3_to_4_t = np.concatenate([np.concatenate([ch.get_kernel(w_128[:, :, ::-1, :][i, j]) for j in range(2)], axis=1) for i in range(2)], axis=0)

# ── JIT filter for samnagui DVS event assembly ──
jit_node = samna.graph.JitFunctionFilter('assembleDvsEvent', '''
        template<class Event>
        auto filterFunction(const Event& input)
        {
            camera::event::DvsEvent event;
            std::visit(
                [&event]<typename T>(const T& e){
                    if constexpr (std::is_same_v<T, ui::DvsEvent>) {
                        int x, y;
                        if (e.layer==13){
                            event.y = e.row;
                            event.x = e.col;
                            event.polarity = e.channel;
                        } else {
                            event.y = e.row * 2 + (e.channel % 2);
                            event.x = e.col * 2 + (e.channel / 2) % 2;
                            event.polarity = ((e.channel / 4) % 2);
                        }
                    }
                },
                input
            );

            return camera::event::DvsEvent{event};
        }
    ''')

# ── CNN 层编号 ──
layer_0_0 = 0
layer_0_1 = 1
layer_1_0 = 7
layer_1_1 = 8
layer_2_0 = 3
layer_2_1 = 4
layer_3_0 = 5
layer_3_1 = 6
layer_4 = 2

config = samna.speck2f.configuration.SpeckConfiguration()


def open_speck2f_dev_kit():
    devices = [
        device
        for device in samna.device.get_unopened_devices()
        if device.device_type_name.startswith("Speck2f")
    ]
    assert devices, "Speck2f board not found"
    default_config = samna.speck2fBoards.DevKitDefaultConfig()
    return samna.device.open_device(devices[0], default_config)


# ===========================================================================
# samnagui 可视化 (from test_online_ring_V1_3.py)
# ===========================================================================

def build_samna_event_route(dk, graph, endpoint, layers):
    """Build a graph in samna to show CNN layer output in samnagui."""
    _, layer_filter, _, _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fOutputMemberSelect",
         "Speck2fDvsToVizConverter", jit_node,
         "CameraToVizConverter", "VizEventStreamer"]
    )
    layer_filter.set_white_list(layers, "layer")

    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint} fails')

    return config_source


def open_visualizer(window_width, window_height, receiver_endpoint):
    """Start visualizer in an isolated process (required on macOS)."""
    gui_process = multiprocessing.Process(
        target=samnagui.run_visualizer,
        args=(receiver_endpoint, window_width, window_height),
    )
    gui_process.start()
    return gui_process


def visualize_layer(dk, layer):
    """Create the lightweight Layer-4 activity view in samnagui."""

    streamer_endpoint = f"tcp://0.0.0.0:4000{layer}"
    gui_process = open_visualizer(0.32, 0.48, streamer_endpoint)
    graph = samna.graph.EventFilterGraph()
    config_source = build_samna_event_route(dk, graph, streamer_endpoint, [layer])
    graph.start()

    plots = [
        samna.ui.ActivityPlotConfiguration(
            128, 128,
            "DVS Layer" if layer == 13 else f"CNN Layer {layer}",
            [0, 0, 1, 1],
        )
    ]
    visualizer_config = samna.ui.VisualizerConfiguration(plots=plots)
    config_source.write([visualizer_config])
    return graph, gui_process


def optimal_sram_config():
    config.factory_config.cnn_layers[0].kernel.clock_pulse = 2
    config.factory_config.cnn_layers[0].kernel.clock_setup = 8
    config.factory_config.cnn_layers[0].kernel.macro_read = 1
    config.factory_config.cnn_layers[0].neuron.clock_pulse = 16
    config.factory_config.cnn_layers[0].neuron.clock_setup = 38
    config.factory_config.cnn_layers[0].neuron.macro_read = 1

    for i in range(1, 9):
        config.factory_config.cnn_layers[i].kernel.clock_pulse = 4
        config.factory_config.cnn_layers[i].kernel.clock_setup = 4
        config.factory_config.cnn_layers[i].kernel.macro_read = 1
        config.factory_config.cnn_layers[i].neuron.clock_pulse = 3
        config.factory_config.cnn_layers[i].neuron.clock_setup = 3
        config.factory_config.cnn_layers[i].neuron.macro_read = 1


def create_layer(layer_name, layer, padding, stride, kernel_size,
                 input_shape_feature, input_shape_size_x, input_shape_size_y,
                 output_shape_feature, output_shape_size_x, output_shape_size_y,
                 threshold_high, threshold_low,
                 weights,
                 destinations_0=None,
                 destinations_1=None,
                 feature_shift_0=None,
                 feature_shift_1=None,
                 monitor_enable=False,
                 leak_enable=False,
                 bias=0):
    print("Create layer:", layer_name)
    dim = samna.speck2f.configuration.CnnLayerDimensions()
    dim.padding.x = padding
    dim.padding.y = padding
    dim.stride.x = stride
    dim.stride.y = stride
    dim.kernel_size = kernel_size
    dim.input_shape.feature_count = input_shape_feature
    dim.input_shape.size.x = input_shape_size_x
    dim.input_shape.size.y = input_shape_size_y
    dim.output_shape.feature_count = output_shape_feature
    dim.output_shape.size.x = output_shape_size_x
    dim.output_shape.size.y = output_shape_size_y
    config.cnn_layers[layer].dimensions = dim
    config.cnn_layers[layer].threshold_high = threshold_high
    config.cnn_layers[layer].threshold_low = threshold_low
    config.cnn_layers[layer].weights = weights
    config.cnn_layers[layer].biases = np.ones(output_shape_feature, dtype=np.int16) * bias
    config.cnn_layers[layer].neurons_initial_value = np.zeros(
        (output_shape_feature, output_shape_size_x, output_shape_size_y), dtype=np.int8)
    config.cnn_layers[layer].leak_enable = leak_enable
    config.cnn_layers[layer].monitor_enable = monitor_enable
    config.cnn_layers[layer].return_to_zero = True
    if destinations_0 is not None:
        config.cnn_layers[layer].destinations[0].layer = destinations_0
        config.cnn_layers[layer].destinations[0].enable = 1
        if feature_shift_0 is not None:
            config.cnn_layers[layer].destinations[0].feature_shift = feature_shift_0
    if destinations_1 is not None:
        config.cnn_layers[layer].destinations[1].layer = destinations_1
        config.cnn_layers[layer].destinations[1].enable = 1
        if feature_shift_1 is not None:
            config.cnn_layers[layer].destinations[1].feature_shift = feature_shift_1


def configure_cnn_pipeline():
    """构建完整的 CNN 流水线配置."""
    config.dvs_layer.destinations[0].layer = layer_0_0
    config.dvs_layer.destinations[0].enable = 1
    config.dvs_layer.destinations[1].layer = layer_0_1
    config.dvs_layer.destinations[1].enable = 1
    optimal_sram_config()

    # ── Layer 0_0 ──
    weights = np.zeros((16, 2, 4, 4), dtype=np.int8)
    for i in range(2):
        for j in range(2):
            weights[j * 2 + i, j, range(i, i + 3), range(i + 2, i - 1, -1)] = [1, 2, 1]
            weights[4 + j * 2 + i, j, range(i, i + 3), range(i + 2, i - 1, -1)] = [1, 2, 1]
            weights[8 + j * 2 + i, j, range(i, i + 3), range(1 - i, 4 - i)] = [1, 2, 1]
            weights[12 + j * 2 + i, j, range(i, i + 3), range(1 - i, 4 - i)] = [1, 2, 1]
    create_layer(
        layer_name="layer_0_0", layer=layer_0_0,
        padding=1, stride=2, kernel_size=4,
        input_shape_feature=2, input_shape_size_x=128, input_shape_size_y=128,
        output_shape_feature=8, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=3, threshold_low=-1,
        weights=weights[:8],
        destinations_0=layer_1_0,
        leak_enable=True, bias=-2,
    )

    # ── Layer 0_1 ──
    create_layer(
        layer_name="layer_0_1", layer=layer_0_1,
        padding=1, stride=2, kernel_size=4,
        input_shape_feature=2, input_shape_size_x=128, input_shape_size_y=128,
        output_shape_feature=8, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=3, threshold_low=-1,
        weights=weights[8:],
        destinations_1=layer_1_1,
        leak_enable=True, bias=-2,
    )

    # ── Layer 1_0 ──
    weights = np.zeros((4, 8, 1, 1), dtype=np.int8)
    for i in range(2):
        weights[i, i, 0, 0] = -1
        weights[i, i + 2, 0, 0] = 2
        weights[i, i + 4, 0, 0] = 1
        weights[i, i + 6, 0, 0] = -2
        weights[i + 2, i, 0, 0] = 2
        weights[i + 2, i + 2, 0, 0] = -1
        weights[i + 2, i + 4, 0, 0] = -2
        weights[i + 2, i + 6, 0, 0] = 1
    create_layer(
        layer_name="layer_1_0", layer=layer_1_0,
        padding=0, stride=1, kernel_size=1,
        input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=4, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_2_0,
        destinations_1=layer_2_0,
        feature_shift_1=4,
    )

    # ── Layer 1_1 ──
    create_layer(
        layer_name="layer_1_1", layer=layer_1_1,
        padding=0, stride=1, kernel_size=1,
        input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=4, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_2_1,
        destinations_1=layer_2_1,
        feature_shift_0=4,
    )

    # ── Layer 2_0 ──
    weights = np.zeros((6, 8, w_1_to_2.shape[2], w_1_to_2.shape[3]), dtype=np.int8)
    weights[:4] = w_1_to_2[np.ix_([0, 3, 4, 7], [0, 3, 0, 3, 4, 7, 4, 7])]
    weights[4, 4::2, (w_1_to_2.shape[2] - 1) // 2, (w_1_to_2.shape[2] - 1) // 2] = 2
    weights[5, 5::2, (w_1_to_2.shape[2] - 1) // 2, (w_1_to_2.shape[2] - 1) // 2] = 2
    create_layer(
        layer_name="layer_2_0", layer=layer_2_0,
        padding=(w_1_to_2.shape[2] - 1) // 2, stride=1, kernel_size=w_1_to_2.shape[2],
        input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=6, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_3_0,
    )

    # ── Layer 2_1 ──
    weights = np.zeros((6, 8, w_1_to_2.shape[2], w_1_to_2.shape[3]), dtype=np.int8)
    weights[:4] = w_1_to_2_t[np.ix_([1, 2, 5, 6], [1, 2, 1, 2, 5, 6, 5, 6])]
    weights[4, 4::2, (w_1_to_2.shape[2] - 1) // 2, (w_1_to_2.shape[2] - 1) // 2] = 2
    weights[5, 5::2, (w_1_to_2.shape[2] - 1) // 2, (w_1_to_2.shape[2] - 1) // 2] = 2
    create_layer(
        layer_name="layer_2_1", layer=layer_2_1,
        padding=(w_1_to_2.shape[2] - 1) // 2, stride=1, kernel_size=w_1_to_2.shape[2],
        input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=6, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_1=layer_3_1,
        feature_shift_0=6,
    )

    # ── Layer 3_0 ──
    weights = np.concatenate([
        w_2_to_3[np.ix_([0, 3, 4, 7], [0, 3, 4, 7])],
        w_0_to_3[np.ix_([0, 3, 4, 7], [0, 3])],
    ], axis=1).astype('int8')
    create_layer(
        layer_name="layer_3_0", layer=layer_3_0,
        padding=(w_2_to_3.shape[2] - 1) // 2, stride=1, kernel_size=w_2_to_3.shape[2],
        input_shape_feature=6, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=4, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_4,
    )

    # ── Layer 3_1 ──
    weights = np.concatenate([
        w_2_to_3_t[np.ix_([1, 2, 5, 6], [1, 2, 5, 6])],
        w_0_to_3_t[np.ix_([1, 2, 5, 6], [1, 2])],
    ], axis=1).astype('int8')
    create_layer(
        layer_name="layer_3_1", layer=layer_3_1,
        padding=(w_2_to_3.shape[2] - 1) // 2, stride=1, kernel_size=w_2_to_3.shape[2],
        input_shape_feature=6, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=4, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_4,
        feature_shift_0=4,
    )

    # ── Layer 4 (输出重排) ──
    # weights = np.zeros((8, 8, 1, 1), dtype=np.int8)
    # weights[0, 0, 0, 0] = 1
    # weights[3, 1, 0, 0] = 1
    # weights[4, 2, 0, 0] = 1
    # weights[7, 3, 0, 0] = 1
    # weights[1, 4, 0, 0] = 1
    # weights[2, 5, 0, 0] = 1
    # weights[5, 6, 0, 0] = 1
    # weights[6, 7, 0, 0] = 1
    # create_layer(
    #     layer_name="layer_4", layer=layer_4,
    #     padding=0, stride=1, kernel_size=1,
    #     input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
    #     output_shape_feature=8, output_shape_size_x=64, output_shape_size_y=64,
    #     threshold_high=1, threshold_low=-1,
    #     weights=weights,
    #     monitor_enable=True,
    # )
    weights = np.zeros((8, 8, 5, 5), dtype=np.int8)
    kernel_anti = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.int8)

    kernel_main = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.int8)

    # 右下、左上：空间方向都是 \
    weights[0, 0] = kernel_main
    weights[3, 1] = kernel_main
    weights[4, 2] = kernel_main
    weights[7, 3] = kernel_main

    # 左下、右上：空间方向都是 /
    weights[1, 4] = kernel_anti
    weights[2, 5] = kernel_anti
    weights[5, 6] = kernel_anti
    weights[6, 7] = kernel_anti

    create_layer(
        layer_name="layer_4", layer=layer_4,
        padding=2, stride=1, kernel_size=5,
        input_shape_feature=8, input_shape_size_x=64, input_shape_size_y=64,
        output_shape_feature=8, output_shape_size_x=64, output_shape_size_y=64,
        threshold_high=4, threshold_low=-1,
        weights=weights,
        monitor_enable=True,
    )

    config.dvs_layer.monitor_enable = False
    config.dvs_layer.pass_sensor_events = True
    config.dvs_layer.mirror.x = True


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
