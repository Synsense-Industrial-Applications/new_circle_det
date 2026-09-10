"""Run real-data integration/Notebook checks and export reviewable previews."""
import argparse
import ast
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import visualize_circle_detection as player_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Reuse matching current-version scores for notebook/layout rechecks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    nb = json.loads((root/"circle_detection_step_by_step.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)
    ns = {}
    exec(code[1], ns)
    ns["PROJECT_DIR"] = args.project
    exec(code[2], ns)
    exec(code[3], ns)
    ns["events"] = ns["events_all"][:1200]
    ns["INSPECT_START_EVENT"], ns["INSPECT_END_EVENT"] = 100, 200
    exec(code[4], ns)
    # Use the exact plotting Cell with the GUI disabled for this headless test.
    exec(code[6].replace("enable_timer=True", "enable_timer=False")
         .replace("plt.show(block=False)", "pass"), ns)
    ns["player"].seek_for_preview(1100)
    ns["player"].figure.canvas.draw()
    plt.close(ns["player"].figure)
    print("Notebook cell execution: PASS", flush=True)

    data, events = ns["data"], ns["events_all"]
    strict_config, adaptive_config = player_module.make_configs()
    started = time.perf_counter()
    signature = player_module._cache_signature(
        ns["CSV_PATH"], len(events), strict_config, adaptive_config)
    cache = root / (
        ns["CSV_PATH"].stem
        + f".circle_cache_v{player_module.CACHE_FORMAT_VERSION}.npz"
    )
    result = (player_module.load_precomputed_cache(cache, signature, len(events))
              if args.reuse_cache else None)
    reused = result is not None
    if result is None:
        result = player_module.precompute_detections(events, strict_config, adaptive_config)
        player_module.save_precomputed_cache(cache, signature, result)
    series = result.adaptive
    full = series.full_confidence
    new = series.xiaoiron_confidence
    finite = np.isfinite(new)
    # Reproducing old geometry/score proves the added score isn't changing the
    # old reference being compared on screen.
    old_path = args.project/(ns["CSV_PATH"].stem+".circle_cache_v2.npz")
    original_equal = None
    if old_path.exists():
        with np.load(old_path) as old:
            original_equal = all(np.allclose(
                getattr(series, key), old["adaptive_"+key], equal_nan=True)
                for key in ("cx", "cy", "radius", "confidence", "support_mode"))
    gain_idx = int(np.nanargmax(new-full))
    loss_idx = int(np.nanargmin(new-full))
    report = dict(
        events=len(events),
        cache_version=player_module.CACHE_FORMAT_VERSION,
        elapsed_seconds=time.perf_counter()-started,
        cache_reused=reused,
        occluded_sectors=list(adaptive_config.xiaoiron.occluded_sectors),
        direction_sectors=list(adaptive_config.xiaoiron.direction_sectors),
        parameters={
            "alpha": adaptive_config.xiaoiron.occlusion_gain,
            "lambda": adaptive_config.xiaoiron.direction_penalty_lambda,
            "N_min": adaptive_config.xiaoiron.min_direction_events,
            "W_min": adaptive_config.xiaoiron.min_direction_weight,
            "K_min": adaptive_config.xiaoiron.min_other_sectors,
            "sector_W": adaptive_config.xiaoiron.min_sector_weight,
        },
        valid_scores=int(finite.sum()),
        increased=int(np.count_nonzero(new > full+1e-6)),
        decreased=int(np.count_nonzero(new < full-1e-6)),
        reward_event=gain_idx + 1,
        reward_full=float(full[gain_idx]),
        reward_score=float(new[gain_idx]),
        penalty_event=loss_idx + 1,
        penalty_full=float(full[loss_idx]),
        penalty_score=float(new[loss_idx]),
        original_equal_v2=original_equal,
    )
    print(json.dumps(report, indent=2), flush=True)
    (root/"xiaoiron_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    ui = player_module.DSCTEventPlayer(
        data, events, result, trail_ms=160, fade_tau_ms=50, interval_ms=30,
        playback_speed=0.01, confidence_threshold=0.18, history_events=600,
        enable_timer=False, xiaoiron_config=adaptive_config.xiaoiron)
    try:
        for label, idx in [("reward", gain_idx), ("penalty", loss_idx)]:
            ui.seek_for_preview(idx)
            ui.figure.canvas.draw()
            renderer = ui.figure.canvas.get_renderer()
            # Main plot and live info must not overlap.
            box = ui.info_text.get_window_extent(renderer)
            assert not box.overlaps(ui.ax_events.get_window_extent(renderer))
            assert not box.overlaps(ui.ax_sector.get_window_extent(renderer))
            ui.figure.savefig(root/f"xiaoiron_{label}_preview.png", dpi=110)
            print(label, "event", idx+1, "full", full[idx], "xiaoiron", new[idx], flush=True)
    finally:
        plt.close(ui.figure)


if __name__ == "__main__":
    main()
