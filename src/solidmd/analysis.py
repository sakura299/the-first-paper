"""Time-origin-averaged tracer MSD and explicitly selected linear fit."""
import math


def msd(positions, species, timestep_fs, selected="Li", max_lag=48):
    atoms = [i for i, symbol in enumerate(species) if symbol == selected]
    if not atoms or not isinstance(max_lag, int) or not 1 <= max_lag < len(positions):
        raise ValueError("Invalid species or MSD lag")
    rows = []
    for lag in range(max_lag + 1):
        origins = len(positions) - lag
        squared = sum(sum((positions[t + lag][atom][k] - positions[t][atom][k]) ** 2
                          for k in range(3)) for t in range(origins) for atom in atoms)
        rows.append({"lag_frames": lag, "time_ps": lag * timestep_fs / 1000,
                     "msd_angstrom2": squared / (origins * len(atoms)),
                     "time_origins": origins})
    return rows


def diffusion_fit(rows, start, stop):
    if not isinstance(start, int) or not isinstance(stop, int) or not 1 <= start < stop <= len(rows):
        raise ValueError("MSD fit uses lag indices [start, stop), with start>=1")
    selected = rows[start:stop]
    x = [row["time_ps"] for row in selected]
    y = [row["msd_angstrom2"] for row in selected]
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    variance = sum((v - xm) ** 2 for v in x)
    if variance == 0:
        raise ValueError("MSD fit needs at least two distinct times")
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / variance
    intercept = ym - slope * xm
    residual = sum((b - slope * a - intercept) ** 2 for a, b in zip(x, y))
    total = sum((b - ym) ** 2 for b in y)
    return {"fit_lag_indices_half_open": [start, stop], "slope_angstrom2_per_ps": slope,
            "intercept_angstrom2": intercept, "r_squared": 1 - residual / total if total else None,
            "D_cm2_per_s": slope / 6 * 1e-4 if slope >= 0 else None,
            "status": "diagnostic_only" if slope >= 0 else "invalid_negative_slope",
            "assumptions": "3D tracer MSD; no drift correction; diffusive regime must be checked manually",
            "uncertainty": "Not estimated; overlapping time origins are correlated"}
