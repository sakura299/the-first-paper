"""Unified schema and conservative fixed orthorhombic PBC handling.

Positions: Angstrom; saved-frame interval: fs; fixed cell, full 3D PBC.
Atom index must refer to the same atom in every frame.
"""
import json
import math
from pathlib import Path


def validate(data):
    if data.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if data.get("units") != {"length": "angstrom", "time": "fs"}:
        raise ValueError("Units must be angstrom and fs")
    dt = data.get("timestep_fs")
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("timestep_fs must be finite and positive")
    if data.get("coordinate_mode") not in ("wrapped", "unwrapped"):
        raise ValueError("coordinate_mode must be wrapped or unwrapped")
    if data.get("pbc") != [True, True, True]:
        raise ValueError("v0.1 requires full 3D periodic boundaries")
    cell = data.get("cell_angstrom", [])
    if len(cell) != 3 or any(len(row) != 3 for row in cell):
        raise ValueError("cell_angstrom must be a 3x3 matrix")
    for a in range(3):
        for b in range(3):
            value = cell[a][b]
            if not math.isfinite(value) or (a == b and value <= 0):
                raise ValueError("Cell must be finite with positive diagonal")
            if a != b and abs(value) > 1e-10:
                raise ValueError("v0.1 supports orthorhombic cells only")
    species = data.get("species", [])
    if not species or any(not isinstance(s, str) or not s for s in species):
        raise ValueError("species must contain nonempty symbols")
    positions = data.get("positions_angstrom", [])
    if len(positions) < 3:
        raise ValueError("At least three frames are required")
    for frame in positions:
        if len(frame) != len(species):
            raise ValueError("Atom count changed")
        for xyz in frame:
            if len(xyz) != 3 or any(not math.isfinite(v) for v in xyz):
                raise ValueError("Coordinates must be finite xyz triplets")
    metadata = data.get("metadata", {})
    if metadata.get("data_kind") not in ("synthetic", "simulation"):
        raise ValueError("metadata.data_kind must be synthetic or simulation")
    if not metadata.get("engine") or not metadata.get("trajectory_id"):
        raise ValueError("Engine and independent trajectory_id are required")
    return data


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate(json.load(handle))


def write(path, data):
    validate(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def unwrap(data):
    """Nearest-image reconstruction: each saved displacement must be < L/2.

    This assumption cannot be recovered or verified from wrapped frames alone.
    Prefer engine-unwrapped coordinates/image flags for sparse saved frames.
    """
    validate(data)
    frames = data["positions_angstrom"]
    if data["coordinate_mode"] == "unwrapped":
        return [[list(xyz) for xyz in frame] for frame in frames]
    lengths = [data["cell_angstrom"][k][k] for k in range(3)]
    result = [[list(xyz) for xyz in frames[0]]]
    for previous, current in zip(frames, frames[1:]):
        frame = []
        for atom, xyz in enumerate(current):
            out = []
            for k, length in enumerate(lengths):
                delta = xyz[k] - previous[atom][k]
                delta -= length * math.floor(delta / length + 0.5)
                if abs(abs(delta) - length / 2) < 1e-10:
                    raise ValueError("Ambiguous half-cell jump; supply unwrapped coordinates")
                out.append(result[-1][atom][k] + delta)
            frame.append(out)
        result.append(frame)
    return result


def synthetic(seed=42, frames=480, timestep_fs=2.0):
    """Correlated Gaussian random walkers, NOT a physical electrolyte model."""
    import random

    if frames < 3:
        raise ValueError("frames must be >=3")
    rng = random.Random(seed)
    species = ["Li"] * 4 + ["O"] * 2
    current = [[rng.uniform(0, 10) for _ in range(3)] for _ in species]
    velocity = [[0.0] * 3 for _ in species]
    positions = [[row[:] for row in current]]
    for _ in range(frames - 1):
        for atom, symbol in enumerate(species):
            for k in range(3):
                sigma = 0.035 if symbol == "Li" else 0.002
                velocity[atom][k] = 0.65 * velocity[atom][k] + rng.gauss(0, sigma)
                current[atom][k] += velocity[atom][k]
        positions.append([row[:] for row in current])
    return validate({
        "schema_version": 1,
        "units": {"length": "angstrom", "time": "fs"},
        "timestep_fs": timestep_fs,
        "cell_angstrom": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [True, True, True],
        "coordinate_mode": "unwrapped",
        "species": species,
        "positions_angstrom": positions,
        "metadata": {"engine": "synthetic_ar1", "data_kind": "synthetic",
                     "trajectory_id": "synthetic-seed-%s" % seed, "seed": seed,
                     "description": "Correlated random walkers; no physical material or temperature"},
    })
