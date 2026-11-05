#!/usr/bin/env python3
"""Import trajectories; coordinates MUST be in angstrom, interval in fs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solidmd.adapters import convert
from solidmd.trajectory import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", choices=["lammps", "dftb+", "vasp", "cp2k"], required=True)
    parser.add_argument("--format", required=True, help="xyz, lammps-dump, or explicit ASE format")
    parser.add_argument("--timestep-fs", type=float, required=True, help="Time BETWEEN SAVED FRAMES")
    parser.add_argument("--coordinate-mode", choices=["wrapped", "unwrapped"], required=True)
    parser.add_argument("--trajectory-id", required=True, help="Independent simulation/group ID")
    parser.add_argument("--box", type=float, nargs=3, help="Plain XYZ fixed cell lengths in angstrom")
    parser.add_argument("--type-map", type=json.loads, help='LAMMPS type map, e.g. \'{"1":"Li","2":"O"}\'')
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    if args.output.exists() and not args.force:
        parser.error("Output exists; choose another path or use --force")
    try:
        data = convert(args.source, args.engine, args.format, args.timestep_fs,
                       args.coordinate_mode, args.trajectory_id, args.box, args.type_map)
        write(args.output, data)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print("Converted %s frames to %s" % (len(data["positions_angstrom"]), args.output))


if __name__ == "__main__":
    main()
