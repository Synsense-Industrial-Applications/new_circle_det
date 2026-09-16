"""Record raw Speck2f CNN Layer-4 events without running circle detection.

The SNN/CNN configuration is imported from the current ``Demo.py`` so this
recorder always uses the same network that is being tested.  Every 16-channel
Layer-4 event is written directly to CSV as ``x,y,feature,timestamp``; no
coordinate decoding, circle fitting, confidence scoring, tracking, or hit
detection is performed.

Run:
    python Demo_record.py

At startup, enter a subfolder name to save this run under
``recordings/<name>/``. Press Enter to save directly under ``recordings/``.

Controls:
    R: start/stop recording.  Each new start creates a new CSV file.
    Q: quit the program.
    Ctrl+C: quit the program safely.
"""

import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import samna

from layer4_layout import LAYER4_FEATURE_COUNT, LAYER4_SOURCE_SIZE

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty

# Import only the hardware/network objects from the user's current Demo.py.
# Importing the module defines the network; it does not call Demo.main().
from Demo import (
    config,
    configure_cnn_pipeline,
    layer_4,
    open_speck2f_dev_kit,
    visualize_layer,
)


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
INVALID_FOLDER_CHARACTERS = frozenset('<>:"/\\|?*')


def choose_recording_directory() -> Path:
    """Ask for one safe subfolder and keep all output under recordings/."""

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    recording_root = RECORD_DIR.resolve()
    while True:
        try:
            folder_name = input(
                "Recording folder name (Enter for recordings/): "
            ).strip()
        except EOFError:
            folder_name = ""
            print("No interactive input; using recordings/.")

        if not folder_name:
            output_dir = recording_root
        elif (
            folder_name in {".", ".."}
            or any(character in INVALID_FOLDER_CHARACTERS for character in folder_name)
            or any(ord(character) < 32 for character in folder_name)
            or folder_name.endswith((" ", "."))
        ):
            print(
                "Invalid folder name. Enter one folder name without path "
                "separators or <>:\"/\\|?*."
            )
            continue
        else:
            output_dir = (RECORD_DIR / folder_name).resolve()

        if output_dir != recording_root and recording_root not in output_dir.parents:
            print("Invalid folder name: output must stay inside recordings/.")
            continue
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"Cannot use that folder: {error}")
            continue
        if not output_dir.is_dir():
            print("Cannot use that name because it is not a directory.")
            continue

        print(f"Recording output directory: {output_dir}")
        return output_dir


def _new_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"layer4_{stamp}.csv"


def _event_row(event):
    """Validate and return one raw 64x64x16 Layer-4 event."""

    row = tuple(int(getattr(event, name)) for name in CSV_COLUMNS)
    x, y, feature, _timestamp = row
    if not 0 <= x < LAYER4_SOURCE_SIZE or not 0 <= y < LAYER4_SOURCE_SIZE:
        raise ValueError(f"Layer-4 coordinates out of range: ({x}, {y})")
    if not 0 <= feature < LAYER4_FEATURE_COUNT:
        raise ValueError(
            f"Layer-4 feature must be in 0..{LAYER4_FEATURE_COUNT - 1}, "
            f"got {feature}"
        )
    return row


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

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.csv_file = None
        self.writer = None
        self.output_path = None
        self.started = 0.0
        self.last_flush = 0.0
        self.events_since_flush = 0
        self.event_count = 0
        self.invalid_count = 0
        self.feature_counts = [0] * LAYER4_FEATURE_COUNT
        self.saved_paths = []

    @property
    def active(self):
        return self.csv_file is not None

    def start(self):
        if self.active:
            return
        self.output_path = _new_output_path(self.output_dir)
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
        self.feature_counts = [0] * LAYER4_FEATURE_COUNT
        print("\n[RECORDING STARTED]")
        print(self.output_path.resolve())

    def write(self, rows, invalid_count=0):
        if not self.active:
            return
        self.invalid_count += invalid_count
        if not rows:
            return
        self.writer.writerows(rows)
        for row in rows:
            self.feature_counts[row[2]] += 1
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
        print(
            "Feature counts: "
            + ", ".join(
                f"{feature}={count:,}"
                for feature, count in enumerate(self.feature_counts)
            )
        )
        print(f"CSV: {output_path.resolve()}")
        print("Press R to create a new recording, or Q to quit.")
        self.csv_file = None
        self.writer = None
        self.output_path = None
        return output_path


def record_layer4():
    """Configure the board and record any number of Layer-4 CSV sessions."""

    output_dir = choose_recording_directory()
    print("Configuring the SNN/CNN pipeline from Demo.py...")
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
    session = CsvRecordingSession(output_dir)
    keyboard = ConsoleKeyReader()

    print(
        f"Logical Layer 4 is ready (hardware layer={layer_4})."
    )
    print("CSV columns: x,y,feature,timestamp")
    print(f"Expected Layer-4 features: 0..{LAYER4_FEATURE_COUNT - 1}")
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
                            f"features_seen="
                            f"{sum(count > 0 for count in session.feature_counts)}/"
                            f"{LAYER4_FEATURE_COUNT} invalid={session.invalid_count:,}"
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
