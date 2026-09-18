"""Speck2f SNN network definition and hardware visualization helpers.

This module contains only the active SNN/CNN configuration: convolution
kernels, layer assignments, device configuration, and samnagui routes.  Circle
recognition and the executable event loop live in ``Demo_algorithm.py``.
Importing this module defines the network configuration but does not open a
device or start a runtime loop.
"""

import multiprocessing

import numpy as np
import samna
import samnagui

from layer4_layout import LAYER4_FEATURE_COUNT, LAYER4_SOURCE_SIZE
from layer4_weights import get_layer4_weights
from speck_tools import ChannelHelper


# ── Layer-4 输出头 ──
# weights 及其配套的 output_features / threshold_high / bias 由 switch（下标）
# 从 layer4_weights.py 取；threshold_low 手动输入。
# 下标 0 是 1x1 直通（8 通道），1..4 输出 16 通道，5..12 是同一套按 0..7 /
# 8..15 切开后的 8 通道版本。
LAYER4_WEIGHT_INDEX = 0        # 见 layer4_weights.format_weights_table()
LAYER4_THRESHOLD_LOW = -1


# Layer-4 输出接口契约：layer4_weights 里每套 weights 都必须产出这个形状，
# 下游 128x128 解码（decode_layer4_address）依赖它。
LAYER4_OUTPUT_SHAPE = (
    LAYER4_SOURCE_SIZE,
    LAYER4_SOURCE_SIZE,
    LAYER4_FEATURE_COUNT,
)


# ===========================================================================
# Speck2f CNN 配置 (from test_online_ring_V1_3.py)
# ===========================================================================

ch = ChannelHelper(input_channel=1, input_size=128, output_size=64)

# ── 构建 optical_flow_split_d_on_off_3_k_conv.py 的最小 2x2 卷积核 ──
# 128 侧图案整体平移后，所有有效 tap 都落在映射核的单侧，因此原来的
# 3x3 粗网格核可以无损裁成 2x2。两级卷积会令输出由 64 缩到 62。
SHIFT_OUT_SIZE = 2


def shift_pattern(pattern, shift_x=1, shift_y=1):
    """平移 128 侧图案，并验证没有非零 tap 被移出边界。"""

    shifted = np.zeros_like(pattern)
    size = pattern.shape[-1]
    for x_index in range(size):
        for y_index in range(size):
            new_x = x_index + shift_x
            new_y = y_index + shift_y
            if 0 <= new_x < size and 0 <= new_y < size:
                shifted[:, :, new_x, new_y] = pattern[:, :, x_index, y_index]
            else:
                assert not np.any(pattern[:, :, x_index, y_index]), (
                    "kernel shift would discard a non-zero tap"
                )
    return shifted


def map_kernel(kernel, size=SHIFT_OUT_SIZE):
    """映射到 64 空间，并使用新网络的最小 2x2 输出核。"""

    mapped = ch.get_kernel(kernel)
    if mapped.shape[-2:] == (size, size):
        return mapped
    if mapped.shape[-2] < size or mapped.shape[-1] < size:
        raise ValueError(
            f"mapped kernel {mapped.shape[-2:]} is smaller than {(size, size)}"
        )

    # shift_pattern() moves all non-zero taps into the lower-right 2x2 part
    # of ChannelHelper's 3x3 result. Verify that cropping is lossless.
    row_start = mapped.shape[-2] - size
    col_start = mapped.shape[-1] - size
    discarded = mapped.copy()
    discarded[..., row_start:, col_start:] = 0
    if np.any(discarded):
        raise ValueError("minimal-kernel crop would discard a non-zero tap")
    return mapped[..., row_start:, col_start:]


def build_64(pattern, transpose=False):
    """镜像可选的 128 侧图案，平移后映射为 2x2 粗网格卷积核。"""

    if transpose:
        pattern = pattern[:, :, ::-1, :]
    pattern = shift_pattern(pattern, 1, 1)
    return np.concatenate(
        [
            np.concatenate(
                [map_kernel(pattern[i, j]) for j in range(pattern.shape[1])],
                axis=1,
            )
            for i in range(pattern.shape[0])
        ],
        axis=0,
    )


M = 5
w_128 = np.zeros((2, 2, M, M))
w_128[1, 1, range(M), range(M)] = [0, -1, 1, -2, 0]
w_128[1, 0, range(M), range(M)] = [0, 0, -1, 2, 0]
w_128[0, 1, range(M), range(M)] = [0, -2, 1, -1, 0]
w_128[0, 0, range(M), range(M)] = [0, 2, -1, 0, 0]
w_1_to_2 = build_64(w_128)
w_1_to_2_t = build_64(w_128, transpose=True)

w_128 = np.zeros((2, 1, M, M))
w_128[1, 0, range(M), range(M)] = [0, 0, 0, -2, 0]
w_128[0, 0, range(M), range(M)] = [0, -2, 0, 0, 0]
w_0_to_3 = build_64(w_128)
w_0_to_3_t = build_64(w_128, transpose=True)

w_128 = np.zeros((2, 2, M, M))
w_128[1, 1, range(M), range(M)] = [0, 1, 2, 0, 0]
w_128[1, 0, range(M), range(M)] = [0, 0, -1, 0, 0]
w_128[0, 1, range(M), range(M)] = [0, 0, -1, 0, 0]
w_128[0, 0, range(M), range(M)] = [0, 0, 2, 1, 0]
w_2_to_3 = build_64(w_128)
w_2_to_3_t = build_64(w_128, transpose=True)

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
    _, event_type_filter, layer_filter, _, _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fOutputEventTypeFilter",
         "Speck2fOutputMemberSelect",
         "Speck2fDvsToVizConverter", jit_node,
         "CameraToVizConverter", "VizEventStreamer"]
    )
    # raw_monitor_enable adds DvsEvent objects to the same output union.  Type
    # filtering first prevents those events (which have no ``layer`` member)
    # from entering the CNN-layer member filter.
    event_type_filter.set_desired_type("speck2f::event::Spike")
    layer_filter.set_white_list(layers, "layer")

    config_source, _ = graph.sequential([samna.BasicSourceNode_ui_event(), streamer])

    streamer.set_streamer_endpoint(endpoint)
    if streamer.wait_for_receiver_count() == 0:
        raise Exception(f'connecting to visualizer on {endpoint} fails')

    return config_source


def build_raw_dvs_event_route(dk, graph, endpoint):
    """Build a samnagui route containing only raw sensor DvsEvent objects."""

    _, event_type_filter, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fOutputEventTypeFilter",
         "Speck2fDvsToVizConverter", "VizEventStreamer"]
    )
    event_type_filter.set_desired_type("speck2f::event::DvsEvent")

    config_source, _ = graph.sequential(
        [samna.BasicSourceNode_ui_event(), streamer]
    )
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


def visualize_raw_dvs(dk):
    """Open a second samnagui window for the raw 128x128 DVS stream."""

    streamer_endpoint = "tcp://0.0.0.0:40013"
    gui_process = open_visualizer(0.32, 0.48, streamer_endpoint)
    graph = samna.graph.EventFilterGraph()
    config_source = build_raw_dvs_event_route(
        dk, graph, streamer_endpoint
    )
    graph.start()

    visualizer_config = samna.ui.VisualizerConfiguration(
        plots=[
            samna.ui.ActivityPlotConfiguration(
                128, 128, "Raw DVS", [0, 0, 1, 1]
            )
        ]
    )
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


def configure_cnn_pipeline(raw_dvs_monitor=False):
    """构建 split-D-ON/OFF 3x3 输入、2x2 映射核的 CNN 流水线。

    Layer-4 输出头不在函数体里硬编码：weights、输出通道数和配套的
    ``threshold_high`` / ``bias`` 由本模块顶部的 ``LAYER4_WEIGHT_INDEX`` 用
    switch 选（见 ``layer4_weights.py``），``LAYER4_THRESHOLD_LOW`` 手动输入。
    """
    # 先取 Layer-4 weights，这样下标写错时会在打开设备之前就报错。
    layer4_head = get_layer4_weights(LAYER4_WEIGHT_INDEX)
    print(
        f"[layer4] weights[{LAYER4_WEIGHT_INDEX}]={layer4_head['name']} "
        f"output_features={layer4_head['output_features']} "
        f"kernel={layer4_head['kernel_size']}x{layer4_head['kernel_size']} "
        f"padding={layer4_head['padding']} "
        f"shape={layer4_head['weights'].shape} "
        f"threshold_high={layer4_head['threshold_high']} "
        f"threshold_low={LAYER4_THRESHOLD_LOW} bias={layer4_head['bias']}"
    )

    config.dvs_layer.destinations[0].layer = layer_0_0
    config.dvs_layer.destinations[0].enable = 1
    # Both split branches must receive the same two-polarity DVS input.  The
    # reference script defines both branches but leaves this route commented;
    # enabling it is required for the S-family (features 1/2/5/6) to emit.
    config.dvs_layer.destinations[1].layer = layer_0_1
    config.dvs_layer.destinations[1].enable = 1
    optimal_sram_config()

    # ── Layer 0_0 ──
    weights = np.zeros((16, 2, 3, 3), dtype=np.int8)
    for i in range(2):
        for j in range(2):
            weights[j * 2 + i, j, i:i + 2, i:i + 2] = 1
            weights[4 + j * 2 + i, j, i:i + 2, i:i + 2] = 1
            weights[8 + j * 2 + i, j, i:i + 2, 1 - i:3 - i] = 1
            weights[12 + j * 2 + i, j, i:i + 2, 1 - i:3 - i] = 1
    create_layer(
        layer_name="layer_0_0", layer=layer_0_0,
        padding=1, stride=2, kernel_size=3,
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
        padding=1, stride=2, kernel_size=3,
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
        feature_shift_1=4,
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
        output_shape_feature=6, output_shape_size_x=63, output_shape_size_y=63,
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
        output_shape_feature=6, output_shape_size_x=63, output_shape_size_y=63,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_1=layer_3_1,
    )

    # ── Layer 3_0 ──
    weights = np.concatenate([
        w_2_to_3[np.ix_([0, 3, 4, 7], [0, 3, 4, 7])],
        w_0_to_3[np.ix_([0, 3, 4, 7], [0, 3])],
    ], axis=1).astype('int8')
    create_layer(
        layer_name="layer_3_0", layer=layer_3_0,
        padding=(w_2_to_3.shape[2] - 1) // 2, stride=1, kernel_size=w_2_to_3.shape[2],
        input_shape_feature=6, input_shape_size_x=63, input_shape_size_y=63,
        output_shape_feature=4, output_shape_size_x=62, output_shape_size_y=62,
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
        input_shape_feature=6, input_shape_size_x=63, input_shape_size_y=63,
        output_shape_feature=4, output_shape_size_x=62, output_shape_size_y=62,
        threshold_high=2, threshold_low=-1,
        weights=weights,
        destinations_0=layer_4,
        feature_shift_0=4,
    )

    # ── Layer 4 (64x64x16 兼容输出头) ──
    # The imported split-D network reaches this head as 62x62x8.  A 3x3
    # convolution with padding=2 expands it to the required 64x64 interface.
    # Padding=2 (not 1) keeps the center tap aligned with the same coarse
    # coordinate frame the previous 5x5/padding=3 head used, so the decoded
    # 128x128 geometry is unchanged while the kernel needs fewer taps.
    # Two eight-channel banks share the optical-flow direction and sub-pixel
    # address mapping; bank 1 uses the complementary spatial diagonal.
    #
    # 权重及配套参数取自本模块顶部的 LAYER4_WEIGHT_INDEX（下标 0 = diag3，
    # 等价于原来的手写权重），threshold_low 手动输入。输出通道数由 weights
    # 条目给出（16 通道整头，或拆开后的 8 通道）。
    create_layer(
        layer_name="layer_4", layer=layer_4,
        padding=layer4_head["padding"], stride=1,
        kernel_size=layer4_head["kernel_size"],
        input_shape_feature=8, input_shape_size_x=62, input_shape_size_y=62,
        output_shape_feature=layer4_head["output_features"],
        output_shape_size_x=LAYER4_SOURCE_SIZE,
        output_shape_size_y=LAYER4_SOURCE_SIZE,
        threshold_high=layer4_head["threshold_high"],
        threshold_low=LAYER4_THRESHOLD_LOW,
        weights=layer4_head["weights"],
        monitor_enable=True,
        leak_enable=True, bias=layer4_head["bias"],
    )

    # ``monitor_enable`` emits pre-processed Layer-13 Spike events, whereas
    # ``raw_monitor_enable`` emits the sensor's original DvsEvent stream.
    # Raw DVS plus Layer-4 monitoring is a high-throughput combination, so use
    # the chip's second output channel only for that opt-in recorder mode.
    config.dvs_layer.monitor_enable = False
    if hasattr(config.dvs_layer, "raw_monitor_enable"):
        config.dvs_layer.raw_monitor_enable = bool(raw_dvs_monitor)
    elif raw_dvs_monitor:
        raise RuntimeError(
            "This samna version does not expose dvs_layer.raw_monitor_enable"
        )
    if hasattr(config.factory_config, "monitor_dual_channel"):
        config.factory_config.monitor_dual_channel = bool(raw_dvs_monitor)
    config.dvs_layer.pass_sensor_events = True
    config.dvs_layer.mirror.x = True

__all__ = (
    "config",
    "configure_cnn_pipeline",
    "layer_4",
    "open_speck2f_dev_kit",
    "visualize_layer",
    "visualize_raw_dvs",
)
