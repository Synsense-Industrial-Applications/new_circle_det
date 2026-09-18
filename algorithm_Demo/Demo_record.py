"""Record Speck2f Layer-4 events and, optionally, raw DVS events.

The SNN/CNN configuration is imported from the current ``Demo_SNN.py`` so this
recorder always uses the same network that is being tested.  Every 16-channel
Layer-4 event is written directly to CSV as ``x,y,feature,timestamp``; no
coordinate decoding, circle fitting, confidence scoring, tracking, or hit
detection is performed. Raw DVS recording is disabled by default. When it is
enabled, a second CSV with ``x,y,polarity,timestamp`` is opened and closed at
exactly the same R-key boundaries as the Layer-4 CSV.

Run:
    python Demo_record.py
    python Demo_record.py --record-dvs
    python Demo_record.py --no-record-dvs

At startup, enter a subfolder name to save this run under
``recordings/<name>/``. Press Enter to save directly under ``recordings/``.
Samnagui always opens both the Layer-4 view and a live raw-DVS view. The DVS
choice controls CSV saving only; it does not disable the live DVS preview.

Controls:
    R: start/stop recording.  Each new start creates a new CSV file.
    Q: quit the program.
    Ctrl+C: quit the program safely.
"""

import argparse
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

# Import only the hardware/network objects from the user's current Demo_SNN.py.
# Importing the module defines the network; it does not start a runtime loop.
from Demo_SNN import (
    config,
    configure_cnn_pipeline,
    layer_4,
    open_speck2f_dev_kit,
    visualize_layer,
    visualize_raw_dvs,
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
RECORDER_INTERFACE_CLOCK_HZ = 25_000_000
CSV_COLUMNS = ("x", "y", "feature", "timestamp")
DVS_CSV_COLUMNS = ("x", "y", "polarity", "timestamp")
DVS_IMAGE_SIZE = 128
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


def choose_dvs_recording() -> bool:
    """Ask whether to save raw DVS events; Enter and EOF both mean no."""

    while True:
        try:
            answer = input("Save raw DVS events to CSV? [y/N]: ").strip().lower()
        except EOFError:
            print("No interactive input; raw DVS recording is disabled.")
            return False
        if answer in ("", "n", "no"):
            return False
        if answer in ("y", "yes"):
            return True
        print("Please enter y or n. Press Enter for the default (no).")


def _new_output_path(output_dir: Path, prefix="layer4", stamp=None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{prefix}_{stamp}.csv"


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


def _first_event_attribute(event, *names):
    for name in names:
        if hasattr(event, name):
            return getattr(event, name)
    raise AttributeError(
        f"{type(event).__name__} has none of the fields {', '.join(names)}"
    )


def _dvs_event_row(event):
    """Return one raw DVS event across supported samna field-name variants."""

    x = int(_first_event_attribute(event, "x", "col"))
    y = int(_first_event_attribute(event, "y", "row"))
    polarity = int(
        _first_event_attribute(event, "p", "polarity", "channel", "feature")
    )
    timestamp = int(_first_event_attribute(event, "timestamp"))
    if not 0 <= x < DVS_IMAGE_SIZE or not 0 <= y < DVS_IMAGE_SIZE:
        raise ValueError(f"raw DVS coordinates out of range: ({x}, {y})")
    if polarity not in (0, 1):
        raise ValueError(f"raw DVS polarity must be 0 or 1, got {polarity}")
    return x, y, polarity, timestamp


def _is_raw_dvs_event(event):
    """Identify ``samna.speck2f.event.DvsEvent`` without assuming one version."""

    event_namespace = getattr(getattr(samna, "speck2f", None), "event", None)
    dvs_event_type = getattr(event_namespace, "DvsEvent", None)
    if dvs_event_type is not None and isinstance(event, dvs_event_type):
        return True
    return type(event).__name__.lower().endswith("dvsevent")


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
    """Own one Layer-4 CSV and an optional, synchronized raw-DVS CSV."""

    def __init__(self, output_dir, record_dvs=False):
        self.output_dir = Path(output_dir)
        self.record_dvs = bool(record_dvs)
        self.csv_file = None
        self.writer = None
        self.output_path = None
        self.dvs_csv_file = None
        self.dvs_writer = None
        self.dvs_output_path = None
        self.started = 0.0
        self.last_flush = 0.0
        self.events_since_flush = 0
        self.event_count = 0
        self.invalid_count = 0
        self.dvs_event_count = 0
        self.dvs_invalid_count = 0
        self.feature_counts = [0] * LAYER4_FEATURE_COUNT
        self.saved_paths = []

    @property
    def active(self):
        return self.csv_file is not None

    def start(self):
        if self.active:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_path = _new_output_path(self.output_dir, "layer4", stamp)
        self.dvs_output_path = (
            _new_output_path(self.output_dir, "dvs", stamp)
            if self.record_dvs
            else None
        )
        self.csv_file = self.output_path.open(
            "w", newline="", encoding="utf-8", buffering=1024 * 1024
        )
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(CSV_COLUMNS)
        self.csv_file.flush()
        if self.dvs_output_path is not None:
            try:
                self.dvs_csv_file = self.dvs_output_path.open(
                    "w", newline="", encoding="utf-8", buffering=1024 * 1024
                )
            except OSError:
                self.csv_file.close()
                self.csv_file = None
                self.writer = None
                self.output_path = None
                self.dvs_output_path = None
                raise
            self.dvs_writer = csv.writer(self.dvs_csv_file)
            self.dvs_writer.writerow(DVS_CSV_COLUMNS)
            self.dvs_csv_file.flush()
        self.started = time.monotonic()
        self.last_flush = self.started
        self.events_since_flush = 0
        self.event_count = 0
        self.invalid_count = 0
        self.dvs_event_count = 0
        self.dvs_invalid_count = 0
        self.feature_counts = [0] * LAYER4_FEATURE_COUNT
        print("\n[RECORDING STARTED]")
        print(f"Layer-4 CSV: {self.output_path.resolve()}")
        if self.dvs_output_path is not None:
            print(f"Raw DVS CSV: {self.dvs_output_path.resolve()}")

    def write(self, rows, dvs_rows=(), invalid_count=0, dvs_invalid_count=0):
        if not self.active:
            return
        self.invalid_count += invalid_count
        self.dvs_invalid_count += dvs_invalid_count
        if rows:
            self.writer.writerows(rows)
            for row in rows:
                self.feature_counts[row[2]] += 1
            written = len(rows)
            self.event_count += written
            self.events_since_flush += written
        if dvs_rows:
            if self.dvs_writer is None:
                raise RuntimeError("received DVS rows while raw DVS recording is off")
            self.dvs_writer.writerows(dvs_rows)
            written = len(dvs_rows)
            self.dvs_event_count += written
            self.events_since_flush += written

    def flush_if_due(self, now):
        if not self.active:
            return
        if (
            self.events_since_flush >= FLUSH_EVERY_EVENTS
            or now - self.last_flush >= FLUSH_INTERVAL_SEC
        ):
            self.csv_file.flush()
            if self.dvs_csv_file is not None:
                self.dvs_csv_file.flush()
            self.events_since_flush = 0
            self.last_flush = now

    def stop(self):
        if not self.active:
            return None
        elapsed = max(time.monotonic() - self.started, 1e-9)
        output_path = self.output_path
        dvs_output_path = self.dvs_output_path
        try:
            self.csv_file.flush()
            if self.dvs_csv_file is not None:
                self.dvs_csv_file.flush()
        finally:
            self.csv_file.close()
            if self.dvs_csv_file is not None:
                self.dvs_csv_file.close()
        self.saved_paths.append(output_path)
        if dvs_output_path is not None:
            self.saved_paths.append(dvs_output_path)
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
        print(f"Layer-4 CSV: {output_path.resolve()}")
        if dvs_output_path is not None:
            print(
                f"Saved {self.dvs_event_count:,} raw DVS events "
                f"({self.dvs_event_count / elapsed:,.1f} event/s); "
                f"invalid={self.dvs_invalid_count:,}"
            )
            print(f"DVS CSV: {dvs_output_path.resolve()}")
        print("Press R to create a new recording, or Q to quit.")
        self.csv_file = None
        self.writer = None
        self.output_path = None
        self.dvs_csv_file = None
        self.dvs_writer = None
        self.dvs_output_path = None
        return output_path


def record_layer4(record_dvs=False):
    """Record Layer-4 sessions and optional synchronized raw-DVS sessions."""

    output_dir = choose_recording_directory()
    print("Configuring the SNN/CNN pipeline from Demo_SNN.py...")
    # Raw monitoring is always required by the live DVS samnagui window.
    # ``record_dvs`` controls only whether those events are also saved to CSV.
    configure_cnn_pipeline(raw_dvs_monitor=True)

    print("Opening Speck2f device...")
    dev_kit = open_speck2f_dev_kit()

    input_graph = samna.graph.EventFilterGraph()
    input_buffer = samna.BasicSourceNode_speck2f_event_input_event()
    input_graph.sequential([input_buffer, dev_kit.get_model_sink_node()])
    input_graph.start()

    dev_kit.get_model().apply_configuration(config)

    # Keep the route alive for the entire recording session.
    device_input_route = samna.graph.source_to(dev_kit.get_model_sink_node())
    recording_output_graph = None
    if record_dvs:
        # The recorder needs both Spike and DvsEvent variants in this mode.
        event_buffer = samna.graph.sink_from(dev_kit.get_model_source_node())
    else:
        # Raw DVS still goes to samnagui, but do not send that high-rate stream
        # through Python when no DVS CSV is requested.
        recording_output_graph = samna.graph.EventFilterGraph()
        (
            _, event_type_filter, layer_filter, event_buffer,
        ) = recording_output_graph.sequential(
            [dev_kit.get_model_source_node(),
             "Speck2fOutputEventTypeFilter", "Speck2fOutputMemberSelect",
             samna.BasicSinkNode_speck2f_event_output_event()]
        )
        event_type_filter.set_desired_type("speck2f::event::Spike")
        layer_filter.set_white_list([layer_4], "layer")
        recording_output_graph.start()

    io_module = dev_kit.get_io_module()
    io_module.set_slow_clk_rate(32)
    io_module.set_slow_clk(True)
    # Both live views are active, so the high-throughput output path is always
    # needed even when raw DVS events are not saved to CSV.
    interface_clock_hz = RECORDER_INTERFACE_CLOCK_HZ
    io_module.set_in_out_interface_clk_rate(interface_clock_hz)
    if hasattr(io_module, "set_dual_channel_output_enable"):
        io_module.set_dual_channel_output_enable(True)
    dev_kit.get_power_module().set_vdd_io(3.3)

    stopwatch = dev_kit.get_stop_watch()
    stopwatch.reset()
    stopwatch.start()

    print("Opening samnagui Layer-4 activity view...")
    visualizers = [visualize_layer(dev_kit, layer_4)]
    print("Opening samnagui raw-DVS activity view...")
    visualizers.append(visualize_raw_dvs(dev_kit))

    # Discard configuration/start-up events before beginning the data file.
    event_buffer.get_events()

    total_raw = 0
    total_layer4_seen = 0
    total_dvs_seen = 0
    last_status = time.monotonic()
    previous_status_seen = 0
    previous_dvs_status_seen = 0
    session = CsvRecordingSession(output_dir, record_dvs=record_dvs)
    keyboard = ConsoleKeyReader()

    print(
        f"Logical Layer 4 is ready (hardware layer={layer_4})."
    )
    print("CSV columns: x,y,feature,timestamp")
    print(f"Expected Layer-4 features: 0..{LAYER4_FEATURE_COUNT - 1}")
    print(f"Raw DVS CSV output: {'ON' if record_dvs else 'OFF'}")
    print("Raw DVS samnagui preview: ON")
    print(f"Output interface clock: {interface_clock_hz / 1_000_000:g} MHz")
    if record_dvs:
        print("DVS CSV columns: x,y,polarity,timestamp")
    print("Dual-channel monitor output: ON")
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
                dvs_rows = []
                invalid_in_batch = 0
                dvs_invalid_in_batch = 0
                for event in events:
                    if getattr(event, "layer", None) == layer_4:
                        total_layer4_seen += 1
                        if not session.active:
                            continue
                        try:
                            rows.append(_event_row(event))
                        except (AttributeError, TypeError, ValueError):
                            invalid_in_batch += 1
                    elif record_dvs and _is_raw_dvs_event(event):
                        total_dvs_seen += 1
                        if not session.active:
                            continue
                        try:
                            dvs_rows.append(_dvs_event_row(event))
                        except (AttributeError, TypeError, ValueError):
                            dvs_invalid_in_batch += 1

                session.write(
                    rows,
                    dvs_rows,
                    invalid_count=invalid_in_batch,
                    dvs_invalid_count=dvs_invalid_in_batch,
                )

                now = time.monotonic()
                session.flush_if_due(now)
                if now - last_status >= STATUS_INTERVAL_SEC:
                    interval = now - last_status
                    input_rate = (
                        total_layer4_seen - previous_status_seen
                    ) / interval
                    dvs_input_rate = (
                        total_dvs_seen - previous_dvs_status_seen
                    ) / interval
                    if session.active:
                        state = (
                            f"RECORDING file_events={session.event_count:,} "
                            f"features_seen="
                            f"{sum(count > 0 for count in session.feature_counts)}/"
                            f"{LAYER4_FEATURE_COUNT} invalid={session.invalid_count:,}"
                        )
                        if record_dvs:
                            state += (
                                f" dvs_events={session.dvs_event_count:,} "
                                f"dvs_invalid={session.dvs_invalid_count:,}"
                            )
                    else:
                        state = "IDLE"
                    rate_text = f"layer4_input={input_rate:,.1f} event/s  "
                    if record_dvs:
                        rate_text += f"dvs_input={dvs_input_rate:,.1f} event/s  "
                    print(
                        f"[{state}] {rate_text}"
                        f"seen={total_layer4_seen:,}  received={total_raw:,}",
                        flush=True,
                    )
                    previous_status_seen = total_layer4_seen
                    previous_dvs_status_seen = total_dvs_seen
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
        if recording_output_graph is not None:
            try:
                recording_output_graph.stop()
            except Exception:
                pass
        for viz_graph, viz_gui in reversed(visualizers):
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Record Layer-4 CSV and optionally a synchronized raw-DVS CSV"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--record-dvs",
        dest="record_dvs",
        action="store_true",
        help="save raw DVS events to a second CSV",
    )
    group.add_argument(
        "--no-record-dvs",
        dest="record_dvs",
        action="store_false",
        help="do not save raw DVS events and skip the startup question",
    )
    parser.set_defaults(record_dvs=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    record_dvs = (
        choose_dvs_recording() if args.record_dvs is None else args.record_dvs
    )
    record_layer4(record_dvs=record_dvs)


if __name__ == "__main__":
    main()
