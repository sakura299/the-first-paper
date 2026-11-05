#!/usr/bin/env python3
"""Generate explicitly synthetic data and run the complete dependency-free demo."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidmd.pipeline import run
from solidmd.trajectory import synthetic, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/demo.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    source = ROOT / config["input"]
    if not args.force and (source.exists() or (output.exists() and any(output.iterdir()))):
        parser.error("Demo data or output already exists; use --force to regenerate")
    write(source, synthetic(config["seed"], config["frames"], config["timestep_fs"]))
    report = run(config, ROOT, force=args.force)
    print("SYNTHETIC DEMO ONLY — no physical material is simulated")
    print("Report:", output / "report.json")
    for name, values in report["metrics"]["test"].items():
        print("%s test RMSE = %.6f A/component" % (name, values["rmse_angstrom_per_component"]))


if __name__ == "__main__":
    main()
