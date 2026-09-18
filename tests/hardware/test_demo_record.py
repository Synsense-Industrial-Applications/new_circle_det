from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ALGORITHM_DIR = PROJECT_ROOT / "algorithm_Demo"
for path in (ALGORITHM_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakeDvsEvent:
    def __init__(self, x, y, p, timestamp):
        self.x = x
        self.y = y
        self.p = p
        self.timestamp = timestamp


def load_recorder_module():
    fake_samna = ModuleType("samna")
    fake_samna.speck2f = SimpleNamespace(
        event=SimpleNamespace(DvsEvent=FakeDvsEvent)
    )
    fake_snn = ModuleType("Demo_SNN")
    fake_snn.config = SimpleNamespace(
        dvs_layer=SimpleNamespace(raw_monitor_enable=False)
    )
    fake_snn.configure_cnn_pipeline = lambda **_kwargs: None
    fake_snn.layer_4 = 2
    fake_snn.open_speck2f_dev_kit = lambda: None
    fake_snn.visualize_layer = lambda *_args: (None, None)
    fake_snn.visualize_raw_dvs = lambda *_args: (None, None)

    spec = importlib.util.spec_from_file_location(
        "Demo_record_under_test", ALGORITHM_DIR / "Demo_record.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"samna": fake_samna, "Demo_SNN": fake_snn},
    ):
        spec.loader.exec_module(module)
    return module


class DemoRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorder = load_recorder_module()

    def test_raw_dvs_field_variants_and_validation(self):
        event = FakeDvsEvent(12, 34, 1, 5678)
        self.assertTrue(self.recorder._is_raw_dvs_event(event))
        self.assertEqual(self.recorder._dvs_event_row(event), (12, 34, 1, 5678))

        alternate = SimpleNamespace(col=5, row=6, channel=0, timestamp=9)
        self.assertEqual(self.recorder._dvs_event_row(alternate), (5, 6, 0, 9))
        with self.assertRaises(ValueError):
            self.recorder._dvs_event_row(FakeDvsEvent(128, 0, 0, 1))
        with self.assertRaises(ValueError):
            self.recorder._dvs_event_row(FakeDvsEvent(0, 0, 2, 1))

    def test_layer4_and_dvs_files_share_one_session_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.recorder.CsvRecordingSession(directory, record_dvs=True)
            session.start()
            layer4_path = session.output_path
            dvs_path = session.dvs_output_path
            self.assertEqual(
                layer4_path.name.removeprefix("layer4_"),
                dvs_path.name.removeprefix("dvs_"),
            )
            session.write(
                [(1, 2, 15, 100)],
                [(3, 4, 1, 101)],
                invalid_count=2,
                dvs_invalid_count=3,
            )
            session.stop()

            with layer4_path.open(newline="", encoding="utf-8") as stream:
                layer4_rows = list(csv.reader(stream))
            with dvs_path.open(newline="", encoding="utf-8") as stream:
                dvs_rows = list(csv.reader(stream))

        self.assertEqual(layer4_rows, [list(self.recorder.CSV_COLUMNS), ["1", "2", "15", "100"]])
        self.assertEqual(dvs_rows, [list(self.recorder.DVS_CSV_COLUMNS), ["3", "4", "1", "101"]])
        self.assertEqual(session.event_count, 1)
        self.assertEqual(session.dvs_event_count, 1)
        self.assertEqual(session.invalid_count, 2)
        self.assertEqual(session.dvs_invalid_count, 3)
        self.assertEqual(session.saved_paths, [layer4_path, dvs_path])

    def test_default_session_creates_only_layer4_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.recorder.CsvRecordingSession(directory)
            session.start()
            session.write([(1, 2, 0, 100)])
            session.stop()
            names = sorted(path.name for path in Path(directory).glob("*.csv"))
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith("layer4_"))

    def test_cli_and_interactive_defaults(self):
        self.assertIsNone(self.recorder.parse_args([]).record_dvs)
        self.assertTrue(self.recorder.parse_args(["--record-dvs"]).record_dvs)
        self.assertFalse(self.recorder.parse_args(["--no-record-dvs"]).record_dvs)
        with patch("builtins.input", return_value=""):
            self.assertFalse(self.recorder.choose_dvs_recording())
        with patch("builtins.input", return_value="y"):
            self.assertTrue(self.recorder.choose_dvs_recording())

    def test_demo_configuration_owns_the_raw_monitor_switch(self):
        source = (ALGORITHM_DIR / "Demo_SNN.py").read_text(encoding="utf-8")
        self.assertIn(
            "def configure_cnn_pipeline(raw_dvs_monitor=False):",
            source,
        )
        self.assertIn(
            "config.dvs_layer.raw_monitor_enable = bool(raw_dvs_monitor)",
            source,
        )
        self.assertIn(
            "config.factory_config.monitor_dual_channel = bool(raw_dvs_monitor)",
            source,
        )
        recorder_source = (ALGORITHM_DIR / "Demo_record.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "configure_cnn_pipeline(raw_dvs_monitor=True)",
            recorder_source,
        )
        self.assertIn("visualize_raw_dvs(dev_kit)", recorder_source)

    def test_recorder_uses_the_high_bandwidth_interface_clock(self):
        self.assertEqual(self.recorder.RECORDER_INTERFACE_CLOCK_HZ, 25_000_000)

    def test_visualization_routes_filter_event_types_before_conversion(self):
        source = (ALGORITHM_DIR / "Demo_SNN.py").read_text(encoding="utf-8")
        self.assertIn(
            'event_type_filter.set_desired_type("speck2f::event::Spike")',
            source,
        )
        self.assertIn(
            'event_type_filter.set_desired_type("speck2f::event::DvsEvent")',
            source,
        )

    def test_split_network_keeps_the_64_by_64_by_16_output_contract(self):
        source = (ALGORITHM_DIR / "Demo_SNN.py").read_text(encoding="utf-8")
        compact = " ".join(source.split())
        self.assertIn("SHIFT_OUT_SIZE = 2", source)
        self.assertIn(
            "padding=1, stride=2, kernel_size=3,",
            source,
        )
        self.assertEqual(
            compact.count(
                "output_shape_feature=6, output_shape_size_x=63, "
                "output_shape_size_y=63,"
            ),
            2,
        )
        self.assertEqual(
            compact.count(
                "output_shape_feature=4, output_shape_size_x=62, "
                "output_shape_size_y=62,"
            ),
            2,
        )
        self.assertIn(
            "padding=2, stride=1, kernel_size=3, "
            "input_shape_feature=8, input_shape_size_x=62, "
            "input_shape_size_y=62,",
            compact,
        )
        self.assertEqual(self.recorder.LAYER4_SOURCE_SIZE, 64)
        self.assertEqual(self.recorder.LAYER4_FEATURE_COUNT, 16)

    def test_snn_and_circle_runtime_are_separate_modules(self):
        snn_source = (ALGORITHM_DIR / "Demo_SNN.py").read_text(encoding="utf-8")
        algorithm_source = (ALGORITHM_DIR / "Demo_algorithm.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("CircleDetectionPipeline", snn_source)
        self.assertNotIn("def main(", snn_source)
        self.assertIn("from Demo_SNN import", algorithm_source)
        self.assertIn("CircleDetectionPipeline", algorithm_source)
        self.assertIn("def main(", algorithm_source)


if __name__ == "__main__":
    unittest.main()
