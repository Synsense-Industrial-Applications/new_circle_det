"""Standalone Speck2f Layer-4 event recorder.

This file contains the complete CNN/SNN configuration, hardware setup,
samnagui Layer-4 visualization, keyboard handling, and CSV recording logic.
It does not depend on Demo.py. The existing speck_tools.py helper is retained.

Runtime dependencies:
    numpy, samna, samnagui, speck_tools.py

Run:
    python Demo_record_pure.py

Controls:
    R: start/stop recording; every start creates a new CSV file.
    Q: quit safely.
    Ctrl+C: quit safely.

CSV output:
    recordings/layer4_YYYYMMDD_HHMMSS_microseconds.csv
    columns: x,y,feature,timestamp
"""

import csv
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import samna
import samnagui

from speck_tools import ChannelHelper

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty


# ===========================================================================
# Built-in Speck2f CNN/SNN configuration
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
        leak_enable=True, bias=-3,
    )

    config.dvs_layer.monitor_enable = False
    config.dvs_layer.pass_sensor_events = True
    config.dvs_layer.mirror.x = True


# ===========================================================================
# Recorder settings
# ===========================================================================

RECORD_DIR = Path(__file__).resolve().parent / "recordings"
READ_BATCH_SIZE = 4096
READ_TIMEOUT_MS = 10
FLUSH_EVERY_EVENTS = 4096
FLUSH_INTERVAL_SEC = 1.0
STATUS_INTERVAL_SEC = 1.0
CSV_COLUMNS = ("x", "y", "feature", "timestamp")
START_STOP_KEY = "r"
QUIT_KEY = "q"


def _new_output_path() -> Path:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return RECORD_DIR / f"layer4_{stamp}.csv"


def _event_row(event):
    """Return one untouched Layer-4 output event in the legacy CSV format."""

    return tuple(int(getattr(event, name)) for name in CSV_COLUMNS)


class ConsoleKeyReader:
    """Non-blocking, no-Enter key input for Windows and Linux terminals."""

    def __init__(self):
        self._stdin_fd = None
        self._saved_terminal_settings = None

    def start(self):
        if os.name == "nt":
            return
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard control requires an interactive terminal (TTY)."
            )
        self._stdin_fd = sys.stdin.fileno()
        self._saved_terminal_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

    def close(self):
        if (
            os.name != "nt"
            and self._stdin_fd is not None
            and self._saved_terminal_settings is not None
        ):
            termios.tcsetattr(
                self._stdin_fd,
                termios.TCSADRAIN,
                self._saved_terminal_settings,
            )
            self._stdin_fd = None
            self._saved_terminal_settings = None

    def read_pressed_keys(self):
        keys = []
        if os.name == "nt":
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    # Consume the second byte of a Windows extended key.
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                keys.append(key.lower())
            return keys

        while select.select([self._stdin_fd], [], [], 0)[0]:
            key_bytes = os.read(self._stdin_fd, 1)
            if not key_bytes:
                break
            keys.append(key_bytes.decode("utf-8", errors="ignore").lower())
        return keys


class CsvRecordingSession:
    """Own one CSV file and its per-recording counters."""

    def __init__(self):
        self.csv_file = None
        self.writer = None
        self.output_path = None
        self.started = 0.0
        self.last_flush = 0.0
        self.events_since_flush = 0
        self.event_count = 0
        self.invalid_count = 0
        self.saved_paths = []

    @property
    def active(self):
        return self.csv_file is not None

    def start(self):
        if self.active:
            return
        self.output_path = _new_output_path()
        self.csv_file = self.output_path.open(
            "w", newline="", encoding="utf-8", buffering=1024 * 1024
        )
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(CSV_COLUMNS)
        self.csv_file.flush()
        self.started = time.monotonic()
        self.last_flush = self.started
        self.events_since_flush = 0
        self.event_count = 0
        self.invalid_count = 0
        print("\n[RECORDING STARTED]")
        print(self.output_path.resolve())

    def write(self, rows, invalid_count=0):
        if not self.active:
            return
        self.invalid_count += invalid_count
        if not rows:
            return
        self.writer.writerows(rows)
        written = len(rows)
        self.event_count += written
        self.events_since_flush += written

    def flush_if_due(self, now):
        if not self.active:
            return
        if (
            self.events_since_flush >= FLUSH_EVERY_EVENTS
            or now - self.last_flush >= FLUSH_INTERVAL_SEC
        ):
            self.csv_file.flush()
            self.events_since_flush = 0
            self.last_flush = now

    def stop(self):
        if not self.active:
            return None
        elapsed = max(time.monotonic() - self.started, 1e-9)
        output_path = self.output_path
        try:
            self.csv_file.flush()
        finally:
            self.csv_file.close()
        self.saved_paths.append(output_path)
        print("\n[RECORDING STOPPED]")
        print(
            f"Saved {self.event_count:,} Layer-4 events in {elapsed:.3f}s "
            f"({self.event_count / elapsed:,.1f} event/s); "
            f"invalid={self.invalid_count:,}"
        )
        print(f"CSV: {output_path.resolve()}")
        print("Press R to create a new recording, or Q to quit.")
        self.csv_file = None
        self.writer = None
        self.output_path = None
        return output_path


def record_layer4():
    """Configure the board and record any number of Layer-4 CSV sessions."""

    print("Configuring the built-in SNN/CNN pipeline...")
    configure_cnn_pipeline()

    print("Opening Speck2f device...")
    dev_kit = open_speck2f_dev_kit()

    input_graph = samna.graph.EventFilterGraph()
    input_buffer = samna.BasicSourceNode_speck2f_event_input_event()
    input_graph.sequential([input_buffer, dev_kit.get_model_sink_node()])
    input_graph.start()

    dev_kit.get_model().apply_configuration(config)

    # Keep the route alive for the entire recording session.
    device_input_route = samna.graph.source_to(dev_kit.get_model_sink_node())
    event_buffer = samna.graph.sink_from(dev_kit.get_model_source_node())

    io_module = dev_kit.get_io_module()
    io_module.set_slow_clk_rate(32)
    io_module.set_slow_clk(True)
    io_module.set_in_out_interface_clk_rate(1_000_000)
    dev_kit.get_power_module().set_vdd_io(3.3)

    stopwatch = dev_kit.get_stop_watch()
    stopwatch.reset()
    stopwatch.start()

    print("Opening samnagui Layer-4 activity view...")
    viz_graph, viz_gui = visualize_layer(dev_kit, layer_4)

    # Discard configuration/start-up events before beginning the data file.
    event_buffer.get_events()

    total_raw = 0
    total_layer4_seen = 0
    last_status = time.monotonic()
    previous_status_seen = 0
    session = CsvRecordingSession()
    keyboard = ConsoleKeyReader()

    print(
        f"Logical Layer 4 is ready (hardware layer={layer_4})."
    )
    print("CSV columns: x,y,feature,timestamp")
    print("Press R to start/stop recording; press Q to quit.")

    try:
        keyboard.start()
        try:
            running = True
            while running:
                for key in keyboard.read_pressed_keys():
                    if key == START_STOP_KEY:
                        if session.active:
                            session.stop()
                        else:
                            # The event stream is drained continuously while idle,
                            # so a new file starts at this key press boundary.
                            session.start()
                    elif key == QUIT_KEY:
                        running = False
                        break

                if not running:
                    break

                events = event_buffer.get_n_events(
                    n=READ_BATCH_SIZE,
                    timeout=READ_TIMEOUT_MS,
                )
                total_raw += len(events)

                rows = []
                invalid_in_batch = 0
                for event in events:
                    if getattr(event, "layer", None) != layer_4:
                        continue
                    total_layer4_seen += 1
                    if not session.active:
                        continue
                    try:
                        rows.append(_event_row(event))
                    except (AttributeError, TypeError, ValueError):
                        invalid_in_batch += 1

                session.write(rows, invalid_in_batch)

                now = time.monotonic()
                session.flush_if_due(now)
                if now - last_status >= STATUS_INTERVAL_SEC:
                    interval = now - last_status
                    input_rate = (
                        total_layer4_seen - previous_status_seen
                    ) / interval
                    if session.active:
                        state = (
                            f"RECORDING file_events={session.event_count:,} "
                            f"invalid={session.invalid_count:,}"
                        )
                    else:
                        state = "IDLE"
                    print(
                        f"[{state}] layer4_input={input_rate:,.1f} event/s  "
                        f"seen={total_layer4_seen:,}  raw={total_raw:,}",
                        flush=True,
                    )
                    previous_status_seen = total_layer4_seen
                    last_status = now

        except KeyboardInterrupt:
            print("\nCtrl+C received; exiting safely...", flush=True)

    finally:
        keyboard.close()
        session.stop()
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
        # Keep the route reference alive until graph shutdown.
        _ = device_input_route

    print(f"Recorder closed. CSV files created: {len(session.saved_paths)}")
    for path in session.saved_paths:
        print(path.resolve())
    return tuple(session.saved_paths)


def main():
    record_layer4()


if __name__ == "__main__":
    main()
