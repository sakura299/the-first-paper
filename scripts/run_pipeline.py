#!/usr/bin/env python3
"""Run from any directory, without installing the package."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidmd.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/local.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run(config, ROOT, force=args.force)
    print(json.dumps({"output": str(ROOT / config["output"]), "test": report["metrics"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
