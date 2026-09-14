"""Record raw Speck2f CNN Layer-4 events without running circle detection.

The SNN/CNN configuration is imported from the current ``Demo.py`` so this
recorder always uses the same network that is being tested.  Every Layer-4
event is written directly to CSV as ``x,y,feature,timestamp``; no coordinate
decoding, circle fitting, confidence scoring, tracking, or hit detection is
performed.

Run:
    python Demo_record.py

Press Ctrl+C to stop.  The output path is printed when recording starts.
"""

import csv
from datetime import datetime
from pathlib import Path
import time

import samna

# Import only the hardware/network objects from the user's current Demo.py.
# Importing the module defines the network; it does not call Demo.main().
from Demo import config, configure_cnn_pipeline, layer_4, open_speck2f_dev_kit


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


def _new_output_path() -> Path:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return RECORD_DIR / f"layer4_{stamp}.csv"


def _event_row(event):
    """Return one untouched Layer-4 output event in the legacy CSV format."""

    return tuple(int(getattr(event, name)) for name in CSV_COLUMNS)


def record_layer4() -> Path:
    """Configure the board and stream Layer-4 output events to one CSV file."""

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

    # Discard configuration/start-up events before beginning the data file.
    event_buffer.get_events()

    output_path = _new_output_path()
    total_raw = 0
    total_layer4 = 0
    invalid_layer4 = 0
    events_since_flush = 0
    started = time.monotonic()
    last_flush = started
    last_status = started
    previous_status_count = 0

    print(
        f"Recording logical Layer 4 output (hardware layer={layer_4}) to:"
    )
    print(output_path.resolve())
    print("CSV columns: x,y,feature,timestamp")
    print("Press Ctrl+C to stop and close the file safely.")

    try:
        with output_path.open(
            "w", newline="", encoding="utf-8", buffering=1024 * 1024
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_COLUMNS)

            try:
                while True:
                    events = event_buffer.get_n_events(
                        n=READ_BATCH_SIZE,
                        timeout=READ_TIMEOUT_MS,
                    )
                    total_raw += len(events)

                    rows = []
                    for event in events:
                        if getattr(event, "layer", None) != layer_4:
                            continue
                        try:
                            rows.append(_event_row(event))
                        except (AttributeError, TypeError, ValueError):
                            invalid_layer4 += 1

                    if rows:
                        writer.writerows(rows)
                        written = len(rows)
                        total_layer4 += written
                        events_since_flush += written

                    now = time.monotonic()
                    if (
                        events_since_flush >= FLUSH_EVERY_EVENTS
                        or now - last_flush >= FLUSH_INTERVAL_SEC
                    ):
                        csv_file.flush()
                        events_since_flush = 0
                        last_flush = now

                    if now - last_status >= STATUS_INTERVAL_SEC:
                        interval = now - last_status
                        rate = (total_layer4 - previous_status_count) / interval
                        print(
                            f"layer4={total_layer4:,}  rate={rate:,.1f} event/s  "
                            f"raw={total_raw:,}  invalid={invalid_layer4:,}",
                            flush=True,
                        )
                        previous_status_count = total_layer4
                        last_status = now

            except KeyboardInterrupt:
                print("\nStop requested; flushing recorded events...", flush=True)

            finally:
                csv_file.flush()

    finally:
        try:
            input_graph.stop()
        except Exception:
            pass
        # Keep the route reference alive until graph shutdown.
        _ = device_input_route

    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"Recording complete: {total_layer4:,} Layer-4 events in "
        f"{elapsed:.3f}s ({total_layer4 / elapsed:,.1f} event/s)."
    )
    print(f"Saved CSV: {output_path.resolve()}")
    return output_path


def main():
    record_layer4()


if __name__ == "__main__":
    main()
