"""Explicit, conservative importers. No engine is executed by these readers."""
import hashlib
from pathlib import Path

from .trajectory import validate


def convert(source, engine, fmt, timestep_fs, coordinate_mode, trajectory_id,
            box=None, type_map=None):
    source = Path(source)
    if fmt == "xyz":
        if box is None:
            raise ValueError("Plain XYZ needs --box Lx Ly Lz in angstrom")
        positions, species = read_xyz(source)
        cell = [[box[k] if k == j else 0.0 for j in range(3)] for k in range(3)]
    elif fmt == "lammps-dump":
        positions, species, cell, detected_mode = read_lammps(source, type_map)
        if coordinate_mode != detected_mode:
            raise ValueError("coordinate-mode disagrees with LAMMPS coordinate columns")
    else:
        try:
            from ase.io import iread
        except ImportError as exc:
            raise ValueError("ASE import requires optional dependency: pip install -e '.[io]'") from exc
        positions, species, cell = [], None, None
        for atoms in iread(str(source), index=":", format=fmt):
            symbols = atoms.get_chemical_symbols()
            matrix = atoms.cell.array.tolist()
            if not all(atoms.pbc):
                raise ValueError("ASE trajectory must contain full PBC and cell information")
            if species is None:
                species, cell = symbols, matrix
            if symbols != species:
                raise ValueError("Species/order changed; stable atom identity is required")
            if any(abs(matrix[i][j] - cell[i][j]) > 1e-8 for i in range(3) for j in range(3)):
                raise ValueError("Variable cells are unsupported in v0.1")
            positions.append(atoms.positions.tolist())
    return validate({
        "schema_version": 1, "units": {"length": "angstrom", "time": "fs"},
        "timestep_fs": timestep_fs, "cell_angstrom": cell, "pbc": [True, True, True],
        "coordinate_mode": coordinate_mode, "species": species,
        "positions_angstrom": positions,
        "metadata": {"engine": engine, "data_kind": "simulation", "trajectory_id": trajectory_id,
                     "source_name": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "format": fmt, "atom_order_assumption": "stable identity across all frames",
                     "time_interval_source": "user-supplied saved-frame interval",
                     "units_assumption": "input positions and cell are in angstrom"},
    })


def read_xyz(source):
    positions, species = [], None
    with Path(source).open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line.strip())
            if count <= 0 or not handle.readline():
                raise ValueError("Invalid XYZ frame")
            symbols, frame = [], []
            for _ in range(count):
                parts = handle.readline().split()
                if len(parts) != 4:
                    raise ValueError("Plain XYZ requires exactly symbol x y z; use ASE for extended XYZ")
                symbols.append(parts[0])
                frame.append([float(v) for v in parts[1:]])
            if species is None:
                species = symbols
            if symbols != species:
                raise ValueError("XYZ atom count/order/species changed")
            positions.append(frame)
    return positions, species


def read_lammps(source, type_map):
    if not type_map:
        raise ValueError("LAMMPS requires an explicit atom type-to-element mapping")
    positions, species, cell, reference_ids, bounds_reference, steps = [], None, None, None, None, []
    mode = None
    with Path(source).open(encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if header.strip() != "ITEM: TIMESTEP":
                raise ValueError("Expected ITEM: TIMESTEP")
            steps.append(int(handle.readline()))
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing atom count")
            count = int(handle.readline())
            if count <= 0:
                raise ValueError("Empty LAMMPS frame")
            if handle.readline().split() != ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"]:
                raise ValueError("LAMMPS reader requires fixed orthorhombic pp pp pp bounds")
            bounds = [[float(v) for v in handle.readline().split()] for _ in range(3)]
            if any(len(row) != 2 for row in bounds):
                raise ValueError("Triclinic bounds are unsupported")
            lengths = [row[1] - row[0] for row in bounds]
            if bounds_reference is None:
                bounds_reference = bounds
                cell = [[lengths[k] if k == j else 0.0 for j in range(3)] for k in range(3)]
            if bounds != bounds_reference:
                raise ValueError("Variable cell/origin is unsupported")
            columns = handle.readline().split()
            if columns[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing atom columns")
            columns = columns[2:]
            coord = next((names for names in [("xu", "yu", "zu"), ("xsu", "ysu", "zsu"),
                                              ("x", "y", "z"), ("xs", "ys", "zs")]
                          if all(name in columns for name in names)), None)
            if coord is None or "id" not in columns or "type" not in columns:
                raise ValueError("LAMMPS needs id, type, and complete x/y/z, xs/ys/zs, xu/yu/zu or xsu/ysu/zsu")
            current_mode = "unwrapped" if coord[0] in ("xu", "xsu") else "wrapped"
            if mode is not None and current_mode != mode:
                raise ValueError("Coordinate convention changed")
            mode = current_mode
            atoms = []
            for _ in range(count):
                parts = handle.readline().split()
                if len(parts) != len(columns):
                    raise ValueError("Truncated atom record")
                record = dict(zip(columns, parts))
                symbol = type_map.get(record["type"])
                if not symbol:
                    raise ValueError("Unmapped atom type: " + record["type"])
                xyz = [float(record[name]) for name in coord]
                if coord[0] in ("xs", "xsu"):
                    xyz = [xyz[k] * lengths[k] for k in range(3)]
                else:
                    xyz = [xyz[k] - bounds[k][0] for k in range(3)]
                atoms.append((int(record["id"]), symbol, xyz))
            atoms.sort(key=lambda row: row[0])
            ids, symbols = [row[0] for row in atoms], [row[1] for row in atoms]
            if len(set(ids)) != count:
                raise ValueError("Duplicate atom IDs")
            if reference_ids is None:
                reference_ids, species = ids, symbols
            if ids != reference_ids or symbols != species:
                raise ValueError("Atom identity/type changed")
            positions.append([row[2] for row in atoms])
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    if gaps and (gaps[0] <= 0 or any(gap != gaps[0] for gap in gaps)):
        raise ValueError("Saved timesteps must be strictly increasing and uniform")
    return positions, species, cell, mode
