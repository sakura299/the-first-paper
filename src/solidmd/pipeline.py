import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .analysis import diffusion_fit, msd
from .features import make_splits, score
from .models import fit_ridge, predict_ridge
from .trajectory import read, unwrap


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config, root, force=False):
    started = time.perf_counter()
    root = Path(root)
    source, output = root / config["input"], root / config["output"]
    if output.exists() and any(output.iterdir()) and not force:
        raise ValueError("Output directory is not empty; choose a new name or use --force")
    data = read(source)
    positions = unwrap(data)
    splits = make_splits(positions, data["species"], config["species"], config["history"],
                         config["train_fraction"], config["validation_fraction"])
    model = fit_ridge(splits["train"]["x"], splits["train"]["y"], config["ridge_alpha"])
    metrics = {}
    for name in ("validation", "test"):
        x, y = splits[name]["x"], splits[name]["y"]
        metrics[name] = {
            "zero_displacement": score([[0.0] * 3 for _ in x], y),
            "last_displacement": score([row[-3:] for row in x], y),
            "ridge": score(predict_ridge(model, x), y),
        }
    curve = msd(positions, data["species"], data["timestep_fs"], config["species"], config["msd_max_lag"])
    fit = diffusion_fit(curve, config["msd_fit_start"], config["msd_fit_stop"])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
        "python": sys.version, "platform": platform.platform(), "git_commit": commit, "git_dirty": dirty,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "config": config,
        "metadata": data["metadata"], "timestep_fs": data["timestep_fs"],
        "representation": "per-ion Cartesian displacement history; no local environment features",
        "split_protocol": "Disjoint contiguous frame blocks; split BEFORE windowing; no shared frames",
        "splits": {key: {"frame_range_half_open": value["frame_range"], "examples": value["examples"]}
                   for key, value in splits.items()},
        "metrics": metrics, "diffusion": fit,
        "limitations": ["Synthetic results are not material properties",
                        "Temporal holdout does not measure cross-trajectory or cross-material generalization",
                        "No equilibration detection, drift removal, conductivity or uncertainty estimation",
                        "Wrapped input requires displacement per saved frame < half the box length"],
        "runtime_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "report.json", report)
    save_json(output / "ridge_model.json", model)
    with (output / "msd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    predictions = predict_ridge(model, splits["test"]["x"])
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example", "true_dx_A", "true_dy_A", "true_dz_A", "pred_dx_A", "pred_dy_A", "pred_dz_A"])
        for i, (target, prediction) in enumerate(zip(splits["test"]["y"], predictions)):
            writer.writerow([i] + target + prediction)
    return report
